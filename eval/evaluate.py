"""체크포인트들을 같은 데이터·같은 지표로 평가한다. 평가 코드는 이 파일과 metrics.py만 쓴다.

  python -m eval.evaluate --data d1 --split test --ckpt checkpoints/d1/m0_s0.pt checkpoints/d1/b1_s0.pt
  python -m eval.evaluate --data d1 --split test --ckpt "checkpoints/d1/*.pt"

결과: eval/results/<data>_<split>.json 과 화면 표
"""
import argparse, glob, json, os
import torch

from data.datasets import load, summary
from eval.metrics import one_step, rollout, HORIZONS
from models.baselines import load_checkpoint
from models.train import data_tag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=25, help="롤아웃 시작점 간격(스텝)")
    ap.add_argument("--rear", action="store_true", help="구간을 뒷바퀴 포화 기준으로 나눔")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    eps = load(args.data, args.split)
    paths = sorted({p for pat in args.ckpt for p in glob.glob(pat)})
    out = dict(data=args.data, split=args.split, summary=summary(eps), regime_axis="rear" if args.rear else "max(front, rear)", models={})
    for p in paths:
        model, ck = load_checkpoint(p)
        name = os.path.relpath(p, ROOT).replace("\\", "/")
        out["models"][name] = dict(kind=ck["kind"], trained_on=ck["info"]["data"],
                                   one_step=one_step(model, eps, model.T, rear=args.rear),
                                   rollout=rollout(model, eps, model.T, stride=args.stride, rear=args.rear),
                                   fitted=ck["info"].get("fitted"))

    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    fn = os.path.join(ROOT, "eval", "results", f"{data_tag(args.data)}_{args.split}{'_rear' if args.rear else ''}.json")
    json.dump(out, open(fn, "w"), indent=1)

    s = out["summary"]
    print(f"\n# {args.data} / {args.split}: 주행 {s['episodes']}개, {s['minutes']}분, μ {s['mu']}, 구간 {s['regime_pct']}")
    print("\n1스텝 nMSE (r, Uy 평균. 작을수록 좋음)")
    print(f"{'model':38s} {'all':>8s} {'normal':>8s} {'nonlin':>8s} {'satur':>8s}")
    for n, m in out["models"].items():
        o = m["one_step"]; g = lambda k: f"{o[k]['mean']:8.4f}" if k in o else f"{'-':>8s}"
        print(f"{n:38s} {g('all')} {g('normal')} {g('nonlinear')} {g('saturated')}")
    print("\n롤아웃 nRMSE (r, Uy 평균. 전체 / 포화 구간)")
    print(f"{'model':38s} " + " ".join(f"{h:>15s}" for h in HORIZONS))
    for n, m in out["models"].items():
        ro = m["rollout"]
        cell = lambda h: f"{ro['all'][h]['mean']:6.3f} / " + (f"{ro['saturated'][h]['mean']:6.3f}" if "saturated" in ro else "     -")
        print(f"{n:38s} " + " ".join(f"{cell(h):>15s}" for h in HORIZONS))
    print("\nsaved", os.path.relpath(fn, ROOT))


if __name__ == "__main__":
    main()
