"""R1~R6 비교 실험: 기준 조합에서 한 가지씩만 바꾼다 (11/17 팀 논의).

  python models/ladder/run_ablation.py train --exp base --rung R2 --seed 0
  python models/ladder/run_ablation.py eval                 # -> eval/results/ladder_ablation.json, 화면 표

실험 (기준 = D7, 힘 입력, 우리 방식 노면(기울기+경사), 1초 손실)
  base       기준
  d1         + D1(GTI, Golf 기준으로 환산) 함께 학습
  pedal      종방향 입력을 페달로
  road_none  노면 없음
  road_bank  팀원 방식 노면(좌우 기울기만, 이득 kb 학습)
  step1      1스텝 손실 (논문식)
  best       각 비교에서 나은 쪽을 합친 조합 (--best 로 지정, 예: --best lon=force road=bank+grade d1=1 H=25)
학습 설정은 팀원 방식: 1스텝 300에폭, 1초 손실은 창 2칸 간격 40에폭(팀원 '세계' 학습과 같음), 마지막 에폭 저장.
평가: D7 시험 주행 3개, 1초·0.5초 롤아웃 RMSE(원단위 vx, vy, r)와 vy·r 정규화 평균, 0.8 g 이상 구간, 현재값 유지 기준선.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import ladder, configs                                   # noqa: E402
from train import fit, windows                           # noqa: E402
from data.ladder_data import d7_segments, d1_segments    # noqa: E402

BASE = dict(lon="force", road="bank+grade", d1=0, H=25, vehicle="team")   # vehicle: 'team' = 11/19 팀 기준 제원·힘 정의
EXPS = dict(base={}, d1=dict(d1=1), pedal=dict(lon="pedal"), road_none=dict(road="none"),
            road_bank=dict(road="bank"), step1=dict(H=1))
CKPT = os.path.join(ROOT, "checkpoints", "ladder")
RUNGS = ["R1", "R2", "R3", "R4", "R5", "R6"]


def spec_of(exp, best=None):
    s = dict(BASE, **EXPS.get(exp, {}))
    for kv in best or []:
        k, v = kv.split("=")
        s[k] = int(v) if k in ("d1", "H") else v
    return s


def legacy(s):
    """11/19 이전 체크포인트 (spec 에 vehicle 키가 없음): 우리 제원(m 1630)과 이전 힘 정의로 불러야 한다."""
    return s.get("vehicle") != "team"


def lon_of(s):
    return "force_old" if legacy(s) and s["lon"] == "force" else s["lon"]


def cfg_of(s):
    return configs.golf(s["lon"], s["road"] == "bank+grade", legacy=legacy(s))


def data_of(s, split):
    segs = d7_segments(split, lon_of(s), s["road"])
    if split == "train" and s["d1"]:
        assert s["lon"] == "force", "D1에는 페달 기록이 없다"
        segs += d1_segments("train", s["road"], m_golf=1630.0 if legacy(s) else 1496.9)
    return segs


def train(args):
    torch.set_num_threads(args.threads)
    s = spec_of(args.exp, args.best)
    cfg = cfg_of(s)
    stride, epochs = (1, 300) if s["H"] == 1 else (2, 40)
    tr = windows(data_of(s, "train"), cfg.hist, horizon=s["H"], stride=stride)
    va = windows(data_of(s, "val"), cfg.hist, horizon=s["H"], stride=stride)
    model, rec = fit(args.rung, cfg, tr, va, epochs=args.epochs or epochs, seed=args.seed, log_every=10, device=args.device)
    model = model.cpu()
    out = os.path.join(CKPT, args.exp, f"{args.rung}_s{args.seed}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ladder.save_model(model, args.rung, out)
    json.dump(dict(spec=s, cfg=cfg.name, rec={k: v for k, v in rec.items() if k != "hist"}, n_train=len(tr[0])),
              open(out[:-3] + ".json", "w"), indent=1)
    print("saved", out)


@torch.no_grad()
def metrics(model, Xt, Yt, Ut, std):
    pred = ladder.rollout(model, Xt, Ut) if model is not None else Xt[:, -1:, :3].expand(-1, Yt.shape[1], -1)
    res = {}
    hi = ((Yt[:, :, 0] * Yt[:, :, 2]).abs().amax(1) / 9.81 > 0.8)       # 1초 안에 0.8 g 이상을 지나는 창
    for name, k in (("0.5s", 12), ("1.0s", 24)):
        e = pred[:, k] - Yt[:, k]
        ok = torch.isfinite(e).all(1)
        raw = e[ok].pow(2).mean(0).sqrt()
        rh = e[ok & hi].pow(2).mean(0).sqrt()
        res[name] = dict(raw=raw.tolist(), n_vyr=float((raw[1:] / std[1:]).mean()), hi_raw=rh.tolist(),
                         diverged=int((~ok).sum()))
    return res


def evaluate(args):
    torch.set_num_threads(args.threads)
    out, cache = {}, {}
    for exp in sorted(os.listdir(CKPT)):
        for p in sorted(glob.glob(os.path.join(CKPT, exp, "*.pt"))):
            meta = json.load(open(p[:-3] + ".json"))
            s = meta["spec"]; cfg = cfg_of(s)
            key = (lon_of(s), s["road"])
            if key not in cache:
                X, Y, U = windows(d7_segments("test", lon_of(s), s["road"]), cfg.hist, horizon=25, stride=5)
                Xt, Yt, Ut = (torch.tensor(z) for z in (X, Y, U))
                std = Yt.reshape(-1, 3).std(0)
                cache[key] = (Xt, Yt, Ut, std)
            Xt, Yt, Ut, std = cache[key]
            rung, seed = os.path.basename(p)[:-3].split("_s")
            model = ladder.load_model(p, cfg)
            out[f"{exp}/{rung}_s{seed}"] = dict(exp=exp, rung=rung, seed=int(seed), spec=s, **metrics(model, Xt, Yt, Ut, std))
            if rung in ("R1", "R2", "R3"):
                v = model.g.values().tolist()
                out[f"{exp}/{rung}_s{seed}"]["coeffs"] = dict(zip(cfg.pnames, v))
    Xt, Yt, Ut, std = cache[next(k for k in cache if k[1] == "bank+grade" and k[0].startswith("force"))]
    out["_hold"] = metrics(None, Xt, Yt, Ut, std)
    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    fn = os.path.join(ROOT, "eval", "results", "ladder_ablation.json")
    json.dump(out, open(fn, "w"), indent=1)

    exps = [e for e in ["base", "d1", "pedal", "road_none", "road_bank", "step1", "best"] if any(r.get("exp") == e for r in out.values() if isinstance(r, dict))]
    print("\nD7 시험, 1초 롤아웃. 셀 = vy·r 정규화 오차 평균 (시드 평균 ± 표준편차) / vx 원단위 [m/s]")
    print(f"{'':10s} " + " ".join(f"{r:>20s}" for r in RUNGS))
    for e in exps:
        cells = []
        for r in RUNGS:
            v = [x for k, x in out.items() if not k.startswith("_") and x["exp"] == e and x["rung"] == r]
            if not v:
                cells.append(f"{'-':>20s}"); continue
            n = np.array([x["1.0s"]["n_vyr"] for x in v]); vx = np.mean([x["1.0s"]["raw"][0] for x in v])
            cells.append(f"{n.mean():6.3f}±{n.std():5.3f} / {vx:5.2f}")
        print(f"{e:10s} " + " ".join(cells))
    h = out["_hold"]["1.0s"]
    print(f"{'현재값 유지':10s} {h['n_vyr']:.3f} / {h['raw'][0]:.2f}")
    print("saved", os.path.relpath(fn, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--exp", default="base")
    ap.add_argument("--rung", default="R1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--best", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu", help="cuda 면 GPU (코랩). R3 1초 손실은 CPU에서 너무 느리다(에폭당 약 15분)")
    args = ap.parse_args()
    (train if args.cmd == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
