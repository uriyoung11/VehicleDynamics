"""시각화 도구 비교용 공용 장면 파일. 어떤 도구든 이 파일 하나만 읽으면 같은 장면을 그릴 수 있다.

  python -m viz.replay --ckpt checkpoints/f1tenth_v3_trial/m0_s0.pt checkpoints/f1tenth_v3_trial/b1_s0.pt
  -> viz/out/replay.npz

내용 (시간축 100 Hz, 길이 n)
  t, X, Y, psi, ux, uy, r, delta, mu, ax, ay      실제 주행 (P2 시뮬레이션)
  margin                                          트랙 가장자리까지 여유 [m] (음수 = 이탈)
  ratio_f, ratio_r, fy_f, fy_r                    타이어 |α|/α_peak, 횡력/(μFz)
  frames                                          예측 시점 인덱스 (20 Hz)
  truth_path (F, 101, 2), path_<model> (F, 101, 2)  각 시점에서 앞으로 1초 궤적: 실제 / 모델 예측
  track_center, track_left, track_right (K, 2), wet_idx   트랙
  car_a, car_b                                    무게중심 ~ 앞/뒤 차축 [m]
"""
import argparse, os
import numpy as np
import torch

from models.baselines import load_checkpoint
from viz.demo_track import predict_paths, truth_paths
from viz.demo_rerun import scenario, Margin, tire_state, P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "viz", "out", "replay.npz")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[os.path.join(ROOT, "checkpoints", "f1tenth_v3_trial", f"{k}_s0.pt") for k in ("m0", "b1")])
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--mu-dry", type=float, default=0.7)
    ap.add_argument("--mu-wet", type=float, default=0.55)
    ap.add_argument("--k-lat", type=float, default=0.75)
    ap.add_argument("--wet-len", type=float, default=25.0)
    ap.add_argument("--before", type=float, default=8.0)
    ap.add_argument("--after", type=float, default=10.0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    tr, wet, sim = scenario(args)
    n = len(sim["t"])
    frames = np.arange(4, n - 101, 5)
    out = {k: sim[k] for k in ("t", "X", "Y", "psi", "ux", "uy", "r", "delta", "mu", "ax", "ay")}
    mfun = Margin(tr)
    out["margin"] = mfun(np.c_[sim["X"], sim["Y"]])[0]
    out.update(tire_state(sim))
    out["frames"] = frames
    out["truth_path"] = truth_paths(sim, frames)
    for p in args.ckpt:
        model, ck = load_checkpoint(p)
        out["path_" + ck["kind"]] = predict_paths(model, sim, frames)
    out["track_center"], out["track_left"], out["track_right"] = tr["center"], tr["left"], tr["right"]
    out["wet_idx"] = np.where((tr["s"] >= wet[0]) & (tr["s"] <= wet[1]))[0]
    out["car_a"], out["car_b"] = P.a, P.b
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print("saved", args.out, f"({n} samples, models: {[k[5:] for k in out if k.startswith('path_')]})")


def load(path=OUT):
    return dict(np.load(path))


if __name__ == "__main__":
    main()
