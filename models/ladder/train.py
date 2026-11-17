"""사다리 모델 학습 (데이터와 무관한 부분만). 지금까지 실험의 src/iac_train.py, src/rv_train.py 와 같은 방식.

  - 손실: 상태별 1스텝 변화량 표준편차로 나눈 MSE (vx·vy·r 단위 차이 보정)
  - roll=1  → 논문식 1스텝 학습
    roll=K  → 기록된 입력으로 K 스텝 굴리며 매 스텝 오차 (k 번째 스텝 오차는 1/k 가중). 우리 실험의 "1초 손실"은 K=25 (dt 0.04)
  - Adam, 전역 물리 계수(GlobalPhys.raw, AxleNet.logf)는 학습률 3배, 코사인 감소, 기울기 노름 10 자름
  - 마지막 에폭 모델을 씀 (검증 최저 에폭은 기록만)

입력 배열 모양 (windows() 로 만듦)
  X (N, H, 6) 이력 창, Y (N, K, 3) 다음 K 스텝 정답 상태, U (N, K-1, 3) 2번째 스텝부터의 입력
"""
import time

import numpy as np
import torch

from ladder import NF, Norm, build


def windows(segments, H, horizon=1, stride=1):
    """segments: [(s (T,3), u (T,3)), ...] 끊김 없는 구간 목록 → (X, Y, U).
    창은 구간 경계를 넘지 않음. 학습·검증·시험은 반드시 '주행(파일) 단위'로 먼저 나눈 뒤 각각 부를 것
    (창을 섞은 뒤 나누면 겹치는 창이 양쪽에 들어가 누수)."""
    X, Y, U = [], [], []
    for s, u in segments:
        su = np.concatenate([s, u], 1)
        for k in range(H - 1, len(su) - horizon, stride):
            X.append(su[k - H + 1:k + 1])
            Y.append(s[k + 1:k + 1 + horizon])
            U.append(u[k + 1:k + horizon])
    nf = segments[0][0].shape[1] + segments[0][1].shape[1] if segments else NF   # [우리 수정] 경사 채널(7)도 받게
    X = np.asarray(X, dtype=np.float32).reshape(-1, H, nf)
    Y = np.asarray(Y, dtype=np.float32).reshape(-1, horizon, 3)
    U = np.asarray(U, dtype=np.float32).reshape(len(X), horizon - 1, nf - 3)
    return X, Y, U


def loss_fn(model, x, y, u, w):
    roll = y.shape[1]
    hist, tot = x, 0.0
    for k in range(roll):
        s = model(hist)
        tot = tot + ((((s - y[:, k]) * w) ** 2) / (k + 1)).mean()
        if k < roll - 1:
            hist = torch.cat([hist[:, 1:], torch.cat([s, u[:, k]], 1).unsqueeze(1)], 1)
    return tot / roll


def fit(rung, cfg, train, val, epochs=300, lr=1e-3, bs=1024, seed=0, device="cpu", log_every=25):
    """train, val: windows() 결과 (X, Y, U). 표준화 상수는 train 에서만 계산. 반환 (model, 기록)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Utr = train
    norm = Norm(Xtr, Ytr)
    model = build(rung, cfg, norm, device)
    T = lambda z: torch.as_tensor(z, dtype=torch.float32, device=device)
    Xtr_t, Ytr_t, Utr_t = T(Xtr), T(Ytr), T(Utr)
    Xva_t, Yva_t, Uva_t = (T(z) for z in val)
    w = 1.0 / norm.dsd.to(device)

    is_fast = lambda n: n.endswith("g.raw") or n.endswith("logf")
    fast = [p for n, p in model.named_parameters() if is_fast(n)]
    rest = [p for n, p in model.named_parameters() if not is_fast(n)]
    groups = [dict(params=rest, lr=lr)] + ([dict(params=fast, lr=3 * lr)] if fast else [])
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    hist, best, best_ep, t0 = [], float("inf"), -1, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = loss_fn(model, Xtr_t[idx], Ytr_t[idx], Utr_t[idx], w)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(loss_fn(model, Xva_t[i:i + 4096], Yva_t[i:i + 4096], Uva_t[i:i + 4096], w).item() * len(Xva_t[i:i + 4096])
                     for i in range(0, len(Xva_t), 4096)) / max(len(Xva_t), 1)
        hist.append((ep, tot / len(perm), vl))
        if vl < best:
            best, best_ep = vl, ep
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"{rung} ep {ep} train {tot / len(perm):.4f} val {vl:.4f} best {best:.4f}@{best_ep} {time.time() - t0:.0f}s", flush=True)
    return model.eval(), dict(hist=hist, best_val=best, best_epoch=best_ep, sec=time.time() - t0)
