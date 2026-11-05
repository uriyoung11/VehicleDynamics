"""모델 학습. 손실은 r, Uy 각각을 표준화한 1스텝 오차(평가의 1스텝 nMSE와 같은 식).

  python -m models.train --data d1 --model m0 --seed 0
  python -m models.train --data d1 --model b1
  python -m models.train --data d2 --model m0
  python -m models.train --data d3:data/generated/f1tenth_v2_trial --model m0

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["m0", "b1"], required=True)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=None, help="기본: m0 1e-3, b1 1e-2")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=30, help="val이 이만큼 안 좋아지면 멈춤")
    ap.add_argument("--tire", default="fiala", help="b1 타이어 모델")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr, va = load(args.data, "train"), load(args.data, "val")
    dt = tr[0]["dt"]
    Xtr, Ytr, _ = windows(tr, args.T); Xva, Yva, _ = windows(va, args.T)
    d_mean, d_std = deriv_stats(Xtr, Ytr, dt)

    if args.model == "m0":
        flat = Xtr.reshape(len(Xtr), -1)
        init = dict(T=args.T, dt=dt, x_mean=flat.mean(0).tolist(), x_std=(flat.std(0) + 1e-8).tolist(),
                    d_mean=d_mean.tolist(), d_std=d_std.tolist())
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
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_state, bad, hist, t0 = np.inf, None, 0, [], time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(); l = loss_fn(model, Xtr[idx], Ytr[idx]).mean(); l.backward(); opt.step()
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
    if args.model == "b1":
        info["fitted"] = model.fitted()
    torch.save(dict(kind=args.model, init=init, state=model.state_dict(), info=info, history=hist), out)
    print("saved", out, json.dumps({k: info[k] for k in ("best_val_nmse", "epochs_run", "seconds")} | ({"fitted": info["fitted"]} if "fitted" in info else {})))


if __name__ == "__main__":
    main()
