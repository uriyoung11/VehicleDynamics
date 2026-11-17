"""썬더힐 GTI 데이터로 R1~R6 이 '돌아가는지'만 보는 짧은 시험 (성능 평가 아님).
  python tests/smoke_thunderhill.py <exp_*.npz 가 있는 폴더> [에폭]
각 단을 몇 에폭만 학습하고, 손실이 유한한지와 25스텝(1초) 굴리기가 발산하지 않는지 확인.
"""
import os
import sys

import numpy as np
import torch

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

import ladder  # noqa: E402
import thunderhill_data as th  # noqa: E402
from configs import GTI_THUNDERHILL as CFG  # noqa: E402
from train import fit, windows  # noqa: E402


def main():
    data_dir = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    torch.set_num_threads(4)
    sp = th.EXAMPLE_SPLIT
    print("파일:", th.available(data_dir))
    segs = {k: th.load_split(data_dir, v) for k, v in sp.items()}
    tr, va = windows(segs["train"], CFG.hist), windows(segs["val"], CFG.hist)
    te = windows(segs["test"], CFG.hist, horizon=25, stride=5)
    print(f"창 수: 학습 {len(tr[0])}, 검증 {len(va[0])}, 시험(25스텝) {len(te[0])}")
    Xte, Yte, Ute = (torch.tensor(z) for z in te)
    hold = (Yte - Xte[:, -1:, :3]).pow(2).mean(dim=(0, 1)).sqrt()
    print("참고: 현재값 유지 25스텝 RMSE vx/vy/r =", np.round(hold.numpy(), 3))
    ok = True
    for rung in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        model, rec = fit(rung, CFG, tr, va, epochs=epochs, log_every=0)
        pred = ladder.rollout(model, Xte, Ute)
        fin = bool(torch.isfinite(pred).all())
        rmse = (pred - Yte).pow(2).mean(dim=(0, 1)).sqrt()
        last = rec["hist"][-1]
        print(f"{rung} {ladder.RUNG_KO[rung]}: 학습 손실 {last[1]:.3f}, 검증 {last[2]:.3f}, "
              f"25스텝 유한 {fin}, RMSE vx/vy/r {np.round(rmse.numpy(), 3)}")
        ok &= fin and np.isfinite(last[1]) and np.isfinite(last[2])
    print("통과" if ok else "실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
