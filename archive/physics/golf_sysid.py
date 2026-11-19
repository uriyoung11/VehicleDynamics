"""D7(Stanford 2025 VW Golf) 데이터로 Golf 물리 시뮬레이터(P2) 파라미터를 추정한다 (시스템 식별).

Golf 제원은 공개 자료에 없다(데이터셋 페이지·연결 논문 미확인, 11/17 검색). 그래서
  고정: 질량 1630 kg(브레이크 힘과 감속도 관계로 추정, data/datasets.py D7_K_BRAKE), 축거 2.63 m(VW Golf 제조사 제원),
        전륜구동, 앞 제동 배분 0.75(D7 바퀴별 브레이크 힘 중앙값)
  추정: 무게중심 위치 a, 요 관성 Iz, 앞뒤 정규화 코너링 강성, Pacejka 형상 C(앞뒤), μ, 무게중심 높이 h,
        주행 저항 c0·c2, 횡속도 센서 위치 보정 ds (Uy_측정 = Uy_무게중심 + ds·r)
손실: 기록된 조향·종방향 힘을 넣고 0.5 s(50스텝) 굴린 [r, Uy, Ux] 오차를 채널별 표준편차로 나눈 MSE.
      한 스텝 손실만 쓰면 물리 계수가 비현실적으로 나온다(우리 M1, 팀원 R1 모두 확인).
입력 Fx: 엔진 토크 × GearRatio × 보정계수 − 바퀴별 브레이크 힘 합 [N] (측정 가속도를 쓰지 않아 정답이 새지 않는다).

  python -m physics.golf_sysid                 # 학습 → physics/golf_fit.json, 화면에 시험 결과
  python -m physics.golf_sysid --eval-only     # 저장된 값으로 시험 주행 검증만
"""
import argparse, glob, json, math, os, time
from dataclasses import replace
import numpy as np
import pandas as pd
import torch

from physics.params import FULLSCALE, TireParams
from physics.bicycle import ModelOptions, rk4_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "physics", "golf_fit.json")
M_GOLF, L_GOLF, BRAKE_FRONT = 1630.0, 2.63, 0.75
DT = 0.01
OPT = ModelOptions(tire="pacejka", load_transfer=True, combined="derate", resistance=True)

# 이름: (초기값, 하한, 상한)
SPEC = {
    "a": (1.05, 0.85, 1.40),          # 무게중심 ~ 앞 차축 [m]. 전륜구동 해치백은 앞쪽 무게 약 60% → a ≈ 0.4 L
    "Iz": (2500.0, 1500.0, 3800.0),   # [kg m²]
    "cf": (15.8, 5.0, 40.0),          # 앞 C_alpha / Fz [1/rad] (GTI 프리셋 15.8)
    "cr": (21.5, 5.0, 50.0),          # 뒤 (GTI 21.5)
    "Cf": (1.6, 1.1, 1.95),           # Pacejka 형상 계수 앞
    "Cr": (1.6, 1.1, 1.95),
    "mu": (1.0, 0.7, 1.5),
    "h": (0.5, 0.3, 0.75),            # 무게중심 높이 [m]
    "c0": (150.0, 0.0, 800.0),        # 구름 저항 [N]
    "c2": (0.4, 0.0, 1.5),            # 공기 저항 계수 [N/(m/s)²]
    "ds": (0.0, -1.5, 1.5),           # 횡속도 센서 위치 보정 [m]
    "kb": (0.0, -1.5, 1.5),           # 좌우 기울기 이득: dUy/dt += g sin(kb·기울기). D7 롤 부호 규약을 몰라 부호·크기를 같이 맞춘다 (--slope일 때만)
}
TRACK = os.path.join(ROOT, "data", "tracks", "ThunderhillWest_centerline.csv")


def slope_lookup():
    """트랙 파일(data.thunderhill_west)의 위치별 [경사, 좌우 기울기]. 위치 [동, 북] → 가장 가까운 트랙 점의 값."""
    from scipy.spatial import cKDTree
    T = np.loadtxt(TRACK, delimiter=",", comments="#")
    assert T.shape[1] >= 7, "트랙 파일에 경사·기울기 열이 없다. python -m data.thunderhill_west 를 먼저 실행"
    tree = cKDTree(T[:, :2])
    return lambda en: T[tree.query(en)[1]][:, 5:7]


def load_runs(split, slope=False):
    """D7 → 연속 구간 리스트. 각 구간 (N, 7) [r, Uy_meas, Ux, delta, Fx_total, 경사, 좌우 기울기] (Ux > 5 m/s).
    slope=False면 경사·기울기 열은 0."""
    look = slope_lookup() if slope else None
    from data.datasets import D7_DIR, D7_DUP, D7_SPLIT, D7_K_DRIVE, D7_K_BRAKE
    segs = []
    for f in sorted(glob.glob(os.path.join(D7_DIR, "*.csv"))):
        name = os.path.basename(f)[:-4]
        if name in D7_DUP:
            continue
        s = next((k for k, runs in D7_SPLIT.items() if name in runs), "train")
        if split not in ("all", s):
            continue
        d = pd.read_csv(f, comment="#")
        cols = ["yawRate_radps", "Vy_mps", "Vx_mps", "roadWheelAngle_rad", "engineTorque_Nm", "GearRatio",
                "brake_fl_N", "brake_fr_N", "brake_rl_N", "brake_rr_N", "posE_m", "posN_m"]
        a = d[cols].to_numpy(np.float64)
        n2 = len(a) // 2 * 2
        a = a[:n2].reshape(-1, 2, len(cols)).mean(1)                 # 100 Hz
        fx = D7_K_DRIVE / D7_K_BRAKE * a[:, 4] * a[:, 5] - a[:, 6:10].sum(1)
        sl = look(a[:, 10:12]) if look else np.zeros((len(a), 2))
        F = np.c_[a[:, 0], a[:, 1], a[:, 2], a[:, 3], fx, sl]
        ok = a[:, 2] > 5.0
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for i0, i1 in zip(edges[::2], edges[1::2]):
            if i1 - i0 > 300:
                segs.append(F[i0:i1])
    return segs


def windows(segs, H, stride):
    W = []
    for F in segs:
        for s in range(0, len(F) - H - 1, stride):
            W.append(F[s:s + H + 1])
    return torch.as_tensor(np.stack(W), dtype=torch.float64)        # (B, H+1, 7)


class GolfParams(torch.nn.Module):
    def __init__(self, init=None):
        super().__init__()
        self.raw = torch.nn.ParameterDict()
        for k, (v0, lo, hi) in SPEC.items():
            v = (init or {}).get(k, v0)
            u = min(max((v - lo) / (hi - lo), 1e-4), 1 - 1e-4)
            self.raw[k] = torch.nn.Parameter(torch.tensor(math.log(u / (1 - u)), dtype=torch.float64))

    def values(self):
        return {k: SPEC[k][1] + (SPEC[k][2] - SPEC[k][1]) * torch.sigmoid(self.raw[k]) for k in SPEC}

    def vehicle(self, v=None):
        v = v or self.values()
        return replace(FULLSCALE, name="golf_d7", m=M_GOLF, Iz=v["Iz"], a=v["a"], b=L_GOLF - v["a"], h=v["h"],
                       tire_f=TireParams(c_alpha=v["cf"], C=v["Cf"]), tire_r=TireParams(c_alpha=v["cr"], C=v["Cr"]),
                       drive_front=1.0, brake_front=BRAKE_FRONT, c0=v["c0"], c1=0.0, c2=v["c2"])


def simulate(P, W, H):
    """W (B, H+1, 5) 기록. 처음 상태에서 시작해 기록된 조향·Fx로 H스텝 굴린다. 반환 (B, H, 3) [r, Uy_meas, Ux]."""
    v = P.values(); p = P.vehicle(v)
    r0, uy0, ux0 = W[:, 0, 0], W[:, 0, 1], W[:, 0, 2]
    z = torch.zeros_like(r0)
    x = torch.stack([z, z, z, ux0, uy0 - v["ds"] * r0, r0], -1)     # 측정 Uy → 무게중심 Uy
    out = []
    for k in range(H):
        u = W[:, k, 3:5]
        slope = torch.stack([W[:, k, 5], v["kb"] * W[:, k, 6]], -1)   # 기록된 경로 위치의 노면 (0.5 s 동안 경로가 크게 안 벗어난다고 봄)
        x = rk4_step(x, u, p, v["mu"], DT, OPT, slope=slope)
        out.append(torch.stack([x[:, 5], x[:, 4] + v["ds"] * x[:, 5], x[:, 3]], -1))
    return torch.stack(out, 1)


def errors(P, segs, horizons=(50, 100, 200), stride=50):
    """시험 지표: 기록 입력으로 굴린 r [rad/s], Uy [m/s], Ux [m/s] RMSE (원단위)."""
    res = {}
    with torch.no_grad():
        for H in horizons:
            W = windows(segs, H, stride)
            pr = simulate(P, W, H) if P is not None else W[:, :1, :3].expand(-1, H, -1)   # None = 현재값 유지
            e = pr[:, -1] - W[:, H, :3]
            ok = torch.isfinite(e).all(1)
            hi = ((W[:, :, 2] * W[:, :, 0]).abs().amax(1) / 9.81 > 0.8) & ok    # 한계 근처(0.8 g 이상) 창
            res[f"{H * DT:.1f}s"] = dict(zip(["r", "uy", "ux"], torch.sqrt((e[ok] ** 2).mean(0)).tolist()),
                                         diverged=int((~ok).sum()),
                                         hi=dict(zip(["r", "uy", "ux"], torch.sqrt((e[hi] ** 2).mean(0)).tolist()), n=int(hi.sum())))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--H", type=int, default=50, help="학습 롤아웃 길이 [스텝]")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--fix", nargs="*", default=[], help="고정할 값, 예: a=1.05 (제조사 무게 배분)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--slope", action="store_true", help="트랙 경사·좌우 기울기의 중력 성분을 넣는다 (kb도 맞춘다)")
    ap.add_argument("--w-limit", type=float, default=0.0, help="구간 안 최대 횡가속도가 0.7 g 이상인 창의 추가 가중치(한계 근처를 더 맞추기)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads); torch.manual_seed(0)
    fixed = {k: float(v) for k, v in (s.split("=") for s in args.fix)}

    tr, va, te = (load_runs(s, args.slope) for s in ("train", "val", "test"))
    if not args.slope:
        fixed.setdefault("kb", 0.0)
    print(f"구간: 학습 {len(tr)}개 {sum(map(len, tr)) / 6000:.1f}분, 검증 {len(va)}개, 시험 {len(te)}개 {sum(map(len, te)) / 6000:.1f}분", flush=True)

    if args.eval_only:
        P = GolfParams(json.load(open(args.out))["values"])
    else:
        Wtr = windows(tr, args.H, 10)
        std = Wtr[:, 1:, :3].reshape(-1, 3).std(0)
        hi = ((Wtr[:, :, 2] * Wtr[:, :, 0]).abs().amax(1) / 9.81 > 0.7).double()
        wtr = 1.0 + args.w_limit * hi
        print(f"0.7 g 이상 창 비율 {float(hi.mean()) * 100:.1f}%, 가중치 {1 + args.w_limit:.1f}", flush=True)
        P = GolfParams(fixed)
        for k in fixed:
            P.raw[k].requires_grad_(False)
        opt = torch.optim.Adam([q for q in P.parameters() if q.requires_grad], lr=args.lr)
        best, best_state, t0 = np.inf, None, time.time()
        Wva = windows(va, args.H, 25)
        for it in range(args.iters):
            idx = torch.randint(len(Wtr), (args.batch,))
            W = Wtr[idx]
            pr = simulate(P, W, args.H)
            loss = (((pr - W[:, 1:, :3]) / std) ** 2).mean((1, 2)) * wtr[idx]
            fin = torch.isfinite(loss)
            loss = loss[fin].sum() / wtr[idx][fin].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 20 == 0 or it == args.iters - 1:
                with torch.no_grad():
                    pv = simulate(P, Wva, args.H)
                    lv = (((pv - Wva[:, 1:, :3]) / std) ** 2)
                    lv = float(lv[torch.isfinite(lv).all(-1).all(-1)].mean())
                if lv < best:
                    best, best_state = lv, {k: t.clone() for k, t in P.state_dict().items()}
                vals = {k: round(float(t), 3) for k, t in P.values().items()}
                print(f"it {it:4d} {time.time() - t0:5.0f}s train {float(loss):.4f} val {lv:.4f}  {vals}", flush=True)
        P.load_state_dict(best_state)
        vals = {k: float(t) for k, t in P.values().items()}
        json.dump(dict(values=vals, fixed=dict(m=M_GOLF, L=L_GOLF, brake_front=BRAKE_FRONT, drive_front=1.0),
                       best_val=best, H=args.H, iters=args.iters, held_fixed=fixed,
                       note="D7 시스템 식별. physics/golf_sysid.py"), open(args.out, "w"), indent=1)
        print("saved", os.path.relpath(args.out, ROOT))

    print("\n시험 주행 롤아웃 RMSE (원단위: r rad/s, Uy m/s, Ux m/s)")
    gti = GolfParams({"a": FULLSCALE.a * L_GOLF / FULLSCALE.L, "Iz": FULLSCALE.Iz, "cf": FULLSCALE.tire_f.c_alpha,
                      "cr": FULLSCALE.tire_r.c_alpha, "mu": 1.0, "h": FULLSCALE.h, "ds": 0.0})
    for name, model in (("Golf 추정", P), ("GTI 제원 그대로", gti), ("현재값 유지", None)):
        res = errors(model, te)
        print(f"{name:14s} " + "  ".join(f"{h}: r {v['r']:.3f} Uy {v['uy']:.3f} Ux {v['ux']:.3f}" for h, v in res.items()))
        print(f"{'  └ 0.8 g 이상':14s} " + "  ".join(f"{h}: r {v['hi']['r']:.3f} Uy {v['hi']['uy']:.3f} (창 {v['hi']['n']}개)" for h, v in res.items()))
    print("\n추정값:", {k: round(float(t), 3) for k, t in P.values().items()}, f"(b = {L_GOLF - float(P.values()['a']):.3f} m)")


if __name__ == "__main__":
    main()
