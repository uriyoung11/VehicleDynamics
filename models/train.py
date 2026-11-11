"""모델 학습. 손실은 r, Uy 각각을 표준화한 1스텝 오차(평가의 1스텝 nMSE와 같은 식).

  python -m models.train --data d1 --model m0 --seed 0
  python -m models.train --data d1 --model b1
  python -m models.train --data d2 --model m0
  python -m models.train --data d3:data/generated/f1tenth_v3_trial --model m0
  python -m models.train --data d1 --model m1        # m1 물리 잔차, m2 신경망 타이어, m3 물리 손실 (models/hybrid.py)

체크포인트: checkpoints/<data 이름>/<model>_s<seed>.pt
"""
import argparse, json, os, time
import numpy as np
import torch

from data.datasets import load, windows, summary
from models.baselines import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_tag(name):
    return name.split(":", 1)[1].rstrip("/").split("/")[-1] if name.startswith("d3:") else name


def deriv_stats(X, Y, dt):
    d = (Y - X[:, -1, :2]) / dt
    return d.mean(0), d.std(0)


def motor_gain_init(X, ep, dt):
    """모터 전류 데이터(D2)의 "구동력 = k · 전류" 초기값. m·dUx/dt ≈ k·I 최소제곱(주행 저항 무시)."""
    if ep["long_kind"] != "motor_current":
        return 1.0
    from physics import PRESETS
    m = PRESETS[ep["vehicle"]].m
    dux = (X[:, -1, 2] - X[:, -2, 2]) / dt
    cur = X[:, -1, 4] * ep["long_scale"]
    k = float((m * dux * cur).sum() / max((cur ** 2).sum(), 1e-9))
    return k if k > 0 else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["m0", "b1", "m1", "m2", "m3"], required=True)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=None, help="기본: b1 1e-2, 나머지 1e-3 (m1~m3의 물리 파라미터는 10배)")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=100, help="val이 이 에폭 수만큼 안 좋아지면 멈춤. D1은 초반 30~40에폭 동안 val이 일시적으로 나빠지므로 짧게 잡지 말 것")
    ap.add_argument("--tire", default="fiala", help="b1 타이어 모델")
    ap.add_argument("--mu-max", type=float, default=1.2, help="m3: 마찰 한계 벌점의 μ 상한 (μ를 모르므로 넉넉하게)")
    ap.add_argument("--lam-c", type=float, default=1.0, help="m3: 제약(마찰 한계·부호) 벌점 가중치")
    ap.add_argument("--lam-m", type=float, default=0.1, help="m3: P1 물리 잔차 벌점 가중치")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr, va = load(args.data, "train"), load(args.data, "val")
    dt = tr[0]["dt"]
    Xtr, Ytr, _ = windows(tr, args.T); Xva, Yva, _ = windows(va, args.T)
    d_mean, d_std = deriv_stats(Xtr, Ytr, dt)

    flat = Xtr.reshape(len(Xtr), -1)
    stats = dict(T=args.T, dt=dt, x_mean=flat.mean(0).tolist(), x_std=(flat.std(0) + 1e-8).tolist(),
                 d_mean=d_mean.tolist(), d_std=d_std.tolist())
    if args.model == "m0":
        init, lr = stats, args.lr or 1e-3
    elif args.model in ("m1", "m2", "m3"):
        init = stats | dict(vehicle=tr[0]["vehicle"], long_kind=tr[0]["long_kind"], long_scale=tr[0]["long_scale"],
                            motor_k0=motor_gain_init(Xtr, tr[0], dt))
        if args.model == "m3":
            init |= dict(mu_max=args.mu_max, lam_c=args.lam_c, lam_m=args.lam_m,
                         state_std=Xtr[:, -1, [0, 1, 3]].std(0).tolist())
        lr = args.lr or 1e-3
    else:
        if tr[0]["long_kind"] != "force_front":
            raise SystemExit("B1은 종방향 입력이 앞바퀴 힘인 데이터(D1, D3)만 지원한다. D2는 모터 전류라 모터 모델이 필요하다.")
        init = dict(vehicle=tr[0]["vehicle"], tire=args.tire, long_scale=tr[0]["long_scale"], dt=dt, T=args.T)
        lr = args.lr or 1e-2
    model = build(args.model, **init).double()

    T_ = lambda a: torch.as_tensor(a, dtype=torch.float64)
    Xtr, Ytr, Xva, Yva = map(T_, (Xtr, Ytr, Xva, Yva))
    scale = T_(dt * d_std)  # 표준화: 미분 오차 / 미분 표준편차
    loss_fn = lambda m, X, Y: (((m(X) - Y) / scale) ** 2).mean(0)
    opt = torch.optim.Adam(model.param_groups(lr) if hasattr(model, "param_groups") else model.parameters(), lr=lr)
    phys_loss = getattr(model, "physics_loss", None)

    best, best_state, bad, hist, t0 = np.inf, None, 0, [], time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            l = loss_fn(model, Xtr[idx], Ytr[idx]).mean()
            if phys_loss is not None:
                l = l + phys_loss(Xtr[idx])
            l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model, Xva, Yva); vm = float(v.mean())
        hist.append(dict(epoch=ep, val=v.tolist()))
        if vm < best - 1e-6:
            best, bad = vm, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0 and ep < 5:
            print(f"[{args.model} {data_tag(args.data)} s{args.seed}] ep {ep:3d} {time.time()-t0:5.0f}s val nMSE r {v[0]:.4f} Uy {v[1]:.4f}", flush=True)
        if bad >= args.patience:
            print(f"early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    out = args.out or os.path.join(ROOT, "checkpoints", data_tag(args.data), f"{args.model}_s{args.seed}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    info = dict(data=args.data, train=summary(tr), val=summary(va), best_val_nmse=best, epochs_run=len(hist),
                seconds=round(time.time() - t0), args=vars(args))
    if hasattr(model, "fitted") and model.fitted():
        info["fitted"] = model.fitted()
    torch.save(dict(kind=args.model, init=init, state=model.state_dict(), info=info, history=hist), out)
    print("saved", out, json.dumps({k: info[k] for k in ("best_val_nmse", "epochs_run", "seconds")} | ({"fitted": info["fitted"]} if "fitted" in info else {})))


if __name__ == "__main__":
    main()
