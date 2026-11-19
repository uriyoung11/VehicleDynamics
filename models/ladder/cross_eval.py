"""차량 간 평가: 사다리 ① base(D7만 학습) vs ② d1(D7+D1 학습)을 D7(Golf) 시험과 D1(GTI) 시험 둘 다에서 잰다.

  python models/ladder/cross_eval.py        # -> eval/results/ladder_cross.json, 화면 표

D1 시험은 d1_segments("test") (Golf 기준으로 힘 환산, 노면 0 — D1 자료에 위치가 없어 경사를 모름).
지표는 run_ablation.metrics 와 같다 (1초 롤아웃, vy·r 정규화 오차 / vx 원단위).
"""
import glob, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import ladder                                             # noqa: E402
from run_ablation import CKPT, RUNGS, cfg_of, metrics     # noqa: E402
from train import windows                                 # noqa: E402
from data.ladder_data import d7_segments, d1_segments     # noqa: E402


def main():
    torch.set_num_threads(8)
    tests = {}
    for name, segs in (("D7", d7_segments("test", "force", "bank+grade")), ("D1", d1_segments("test", "bank+grade"))):
        X, Y, U = windows(segs, 8, horizon=25, stride=5)
        Xt, Yt, Ut = (torch.tensor(z) for z in (X, Y, U))
        tests[name] = (Xt, Yt, Ut, Yt.reshape(-1, 3).std(0))
        print(name, "시험 창", len(Xt))
    out = {f"_hold/{t}": metrics(None, *v) for t, v in tests.items()}
    for exp in ("base", "d1"):
        for p in sorted(glob.glob(os.path.join(CKPT, exp, "*.pt"))):
            s = json.load(open(p[:-3] + ".json"))["spec"]
            cfg = cfg_of(s)
            rung, seed = os.path.basename(p)[:-3].split("_s")
            model = ladder.load_model(p, cfg)
            for t, v in tests.items():
                out[f"{exp}/{rung}_s{seed}/{t}"] = dict(exp=exp, rung=rung, seed=int(seed), test=t, **metrics(model, *v))
    fn = os.path.join(ROOT, "eval", "results", "ladder_cross.json")
    json.dump(out, open(fn, "w"), indent=1)

    print("\n1초 롤아웃. 셀 = vy·r 정규화 오차 (시드 평균 ± 표준편차) / vx [m/s] / 발산 창 수")
    for t in tests:
        h = out[f"_hold/{t}"]["1.0s"]
        print(f"\n[{t} 시험]  현재값 유지 {h['n_vyr']:.3f} / {h['raw'][0]:.2f}")
        print(f"{'':6s} " + " ".join(f"{r:>24s}" for r in RUNGS))
        for exp in ("base", "d1"):
            cells = []
            for r in RUNGS:
                v = [x for k, x in out.items() if not k.startswith("_") and x["exp"] == exp and x["rung"] == r and x["test"] == t]
                if not v:
                    cells.append(f"{'-':>24s}"); continue
                n = np.array([x["1.0s"]["n_vyr"] for x in v])
                vx = np.mean([x["1.0s"]["raw"][0] for x in v])
                dv = sum(x["1.0s"]["diverged"] for x in v)
                cells.append(f"{n.mean():6.3f}±{n.std():5.3f} / {vx:5.2f} / {dv}")
            print(f"{exp:6s} " + " ".join(cells))
    print("saved", os.path.relpath(fn, ROOT))


if __name__ == "__main__":
    main()
