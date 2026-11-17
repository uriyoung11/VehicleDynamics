"""Golf/D7 설정으로 R1~R6이 '돌아가는지'만 보는 짧은 시험 (성능 평가 아님).

  python models/ladder/tests/smoke_golf.py [에폭]

조합: (힘, 노면 없음), (힘, 기울기), (힘, 기울기+경사), (페달, 기울기+경사), (힘, 기울기+경사, D1 함께).
각 R1~R6을 몇 에폭 학습하고, 1스텝 손실과 1초(25스텝) 손실 학습 둘 다 유한한지, 25스텝 굴리기가 유한한지 본다.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import ladder, configs                       # noqa: E402
from train import fit, windows               # noqa: E402
from data.ladder_data import d7_segments, d1_segments   # noqa: E402

COMBOS = [("force", "none", False), ("force", "bank", False), ("force", "bank+grade", False),
          ("pedal", "bank+grade", False), ("force", "bank+grade", True)]


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    torch.set_num_threads(8)
    ok = True
    for lon, road, with_d1 in COMBOS:
        cfg = configs.golf(lon, road == "bank+grade")
        tr = d7_segments("train", lon, road) + (d1_segments("train", road) if with_d1 else [])
        va, te = d7_segments("val", lon, road), d7_segments("test", lon, road)
        Xte, Yte, Ute = (torch.tensor(z) for z in windows(te, cfg.hist, horizon=25, stride=10))
        tag = f"{lon}/{road}{' +D1' if with_d1 else ''}"
        for rung in ("R1", "R2", "R3", "R4", "R5", "R6"):
            for H in (1, 25):
                trw = windows(tr, cfg.hist, horizon=H, stride=1 if H == 1 else 4)
                vaw = windows(va, cfg.hist, horizon=H, stride=1 if H == 1 else 4)
                model, rec = fit(rung, cfg, trw, vaw, epochs=epochs, log_every=0)
                pred = ladder.rollout(model, Xte, Ute)
                fin = bool(torch.isfinite(pred).all()) and np.isfinite(rec["hist"][-1][1])
                rm = (pred[:, -1] - Yte[:, -1]).pow(2).mean(0).sqrt().numpy()
                print(f"{tag:22s} {rung} 손실 {'1스텝' if H == 1 else '1초 '}: 유한 {fin}, 1초 RMSE vx/vy/r {np.round(rm, 3)}", flush=True)
                ok &= fin
    print("통과" if ok else "실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
