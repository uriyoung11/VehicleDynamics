"""여러 차 데이터 혼합 실험 평가: checkpoints/mix/<학습 데이터>/<모델>_s<시드>.pt 를 D1(GTI)·D7(Golf) 시험 데이터에서 평가한다.

  python -m eval.mix_eval                      # -> eval/results/mixing.json, 화면 표
  python -m eval.mix_eval --root checkpoints/mix --out eval/results/mixing.json

학습 데이터 이름(폴더)
  d1        GTI만          d7        Golf만
  d1+d7     그냥 섞기       d1+d7_veh 섞고 차 구분 채널 추가 (평가도 차 구분 채널을 붙인 데이터로)
지표: 1스텝 nMSE, 롤아웃 nRMSE(0.5 s, 1.0 s). eval/metrics.py 그대로.
"""
import argparse, glob, json, os
import numpy as np
import torch

from data.datasets import load
from eval.metrics import one_step, rollout, Hold
from models.baselines import load_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SETS = ["d1", "d7", "d1+d7", "d1+d7_veh"]
TEST_SETS = ["d1", "d7"]
MODELS = ["b1", "m0", "m1", "m2", "m3"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "checkpoints", "mix"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results", "mixing.json"))
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    tests = {(t, veh): load(t + ("|veh" if veh else ""), "test") for t in TEST_SETS for veh in (False, True)}
    res = {}
    for tr in TRAIN_SETS:
        veh = tr.endswith("_veh")
        for p in sorted(glob.glob(os.path.join(args.root, tr, "*.pt"))):
            model, ck = load_checkpoint(p)
            key = f"{tr}/{os.path.basename(p)[:-3]}"
            res[key] = dict(train=tr, kind=ck["kind"], seed=int(os.path.basename(p)[:-3].split("_s")[-1]), test={})
            for t in TEST_SETS:
                eps = tests[(t, veh)]
                o = one_step(model, eps, model.T)
                ro = rollout(model, eps, model.T, stride=args.stride)
                res[key]["test"][t] = dict(one_step=o["all"]["mean"],
                                          r05=ro["all"]["0.5s"]["mean"], r10=ro["all"]["1.0s"]["mean"],
                                          r10_nonlin=ro.get("nonlinear", {}).get("1.0s", {}).get("mean"),
                                          r10_sat=ro.get("saturated", {}).get("1.0s", {}).get("mean"),
                                          raw10=ro["all"]["1.0s"]["raw"],
                                          diverged=ro["all"]["1.0s"]["diverged"])
            print(key, {t: round(v["r10"], 3) for t, v in res[key]["test"].items()}, flush=True)
    # 기준선: 현재값 유지(학습 없음)
    hold = {t: rollout(Hold(4), tests[(t, False)], 4, stride=args.stride)["all"] for t in TEST_SETS}
    res["_hold"] = {t: dict(r05=h["0.5s"]["mean"], r10=h["1.0s"]["mean"], raw10=h["1.0s"]["raw"]) for t, h in hold.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)

    # 표: 학습 데이터 × 모델, 셀 = 시드 평균 1초 롤아웃 nRMSE (GTI 시험 / Golf 시험)
    print("\n1초 롤아웃 nRMSE, 시드 평균 (GTI 시험 / Golf 시험). 작을수록 좋음")
    print(f"{'학습':12s} " + " ".join(f"{m:>15s}" for m in MODELS))
    for tr in TRAIN_SETS:
        cells = []
        for m in MODELS:
            v = [r["test"] for k, r in res.items() if not k.startswith("_") and r["train"] == tr and r["kind"] == m]
            if not v:
                cells.append(f"{'-':>15s}"); continue
            a = np.mean([x["d1"]["r10"] for x in v]); b = np.mean([x["d7"]["r10"] for x in v])
            cells.append(f"{a:6.3f} / {b:6.3f}")
        print(f"{tr:12s} " + " ".join(cells))
    h = res["_hold"]
    print(f"{'현재값 유지':12s} {h['d1']['r10']:6.3f} / {h['d7']['r10']:6.3f}   (원단위 1초: GTI r {h['d1']['raw10'][0]:.3f} rad/s Uy {h['d1']['raw10'][1]:.3f} m/s)")
    print("saved", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
