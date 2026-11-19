"""입출력 형식 비교 평가: checkpoints/fmt/<형식>/<모델>_s<시드>.pt 를 Golf(D7)·GTI(D1) 시험 데이터에서 평가한다.

  python -m eval.format_eval            # -> eval/results/formats.json, 화면 표

시험 데이터는 학습 데이터 이름에서 정한다(간격 @k는 그대로):
  "d1+d7a@2" → Golf "d7a@2", GTI "d1@2".  "d7p@2" → Golf "d7p@2" (GTI 없음)
지표는 eval/metrics.py: 1초·0.5초 롤아웃 nRMSE(r, Uy 평균. 간격이 달라도 같은 1초), 원단위 RMSE, Ux 오차(출력이 3개일 때).
"""
import argparse, glob, json, os, re
import numpy as np
import torch

from data.datasets import load
from eval.metrics import rollout, Hold
from models.baselines import load_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMATS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7a", "F7p"]
MODELS = ["b1", "m0", "m1"]


def tests_for(train_name):
    base, k = (train_name.split("@") + [None])[:2]
    sfx = f"@{k}" if k else ""
    golf = ("d7p" if "d7p" in base else "d7a") + sfx
    return {"golf": golf, **({"gti": "d1" + sfx} if base.startswith("d1") else {})}


def summarize(ro):
    a = ro["all"]
    out = dict(r05=a["0.5s"]["mean"], r10=a["1.0s"]["mean"], raw10=a["1.0s"]["raw"], diverged=a["1.0s"]["diverged"])
    if "ux" in a["1.0s"]:
        out["ux10"] = a["1.0s"]["ux"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "checkpoints", "fmt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results", "formats.json"))
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    cache, res = {}, {}
    for F in FORMATS:
        for p in sorted(glob.glob(os.path.join(args.root, F, "*.pt"))):
            model, ck = load_checkpoint(p)
            tn = tests_for(ck["info"]["data"])
            key = f"{F}/{os.path.basename(p)[:-3]}"
            res[key] = dict(format=F, kind=ck["kind"], seed=int(re.search(r"_s(\d+)", p).group(1)), T=model.T, dt=model.dt,
                            n_out=getattr(model, "n_out", 2), train=ck["info"]["data"], test={})
            for which, name in tn.items():
                if name not in cache:
                    cache[name] = load(name, "test")
                res[key]["test"][which] = summarize(rollout(model, cache[name], model.T))
            print(key, {w: round(v["r10"], 3) for w, v in res[key]["test"].items()}, flush=True)
    res["_hold"] = {name: summarize(rollout(Hold(4), eps, 4)) for name, eps in cache.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)

    print("\n1초 롤아웃 nRMSE (r, Uy 평균), 시드 평균: Golf 시험 / GTI 시험. 작을수록 좋음")
    print(f"{'형식':6s} " + " ".join(f"{m:>17s}" for m in MODELS))
    for F in FORMATS:
        cells = []
        for m in MODELS:
            v = [r for k, r in res.items() if not k.startswith("_") and r["format"] == F and r["kind"] == m]
            if not v:
                cells.append(f"{'-':>17s}"); continue
            g = np.mean([x["test"]["golf"]["r10"] for x in v])
            t = np.mean([x["test"]["gti"]["r10"] for x in v]) if "gti" in v[0]["test"] else np.nan
            cells.append(f"{g:7.3f} / {t:7.3f}")
        print(f"{F:6s} " + " ".join(cells))
    print("현재값 유지:", {k: round(v["r10"], 3) for k, v in res["_hold"].items()})
    print("saved", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
