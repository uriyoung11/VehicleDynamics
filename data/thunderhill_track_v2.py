"""Thunderhill West 트랙 v2: 우리 트랙(OSM 노면 중앙선 + 노면 지도)을 기반으로 팀원 트랙의 좋은 점을 차용 (11/19 팀 결정).

  python -m data.thunderhill_track_v2            # -> data/tracks/v2/thunderhill_{center,mincurv,recorded}.npz, info.json, 그림

우리 것에서 그대로 가져온 것 (data/thunderhill_west.py, data/tracks/ThunderhillWest_centerline.csv)
  - 경계의 위치: OSM 노면 중앙선(D7 레이싱 라인으로 꺾임만 다듬음). 레이싱 라인은 코너 안·바깥 끝을 지나므로
    주행선을 기준으로 경계를 두면 코너 안쪽이 실제 잔디·연석까지 열린다.
  - 노면: 고도, 경사(GPS 고도를 트랙 거리로 미분), 좌우 기울기(믿기 어려워 참고용).
팀원 것(Kxngstonx/physics_for_driving, deep_dynamics/golf/track.py)에서 차용한 것
  - 폭: Google Earth로 West 14곳을 3번씩 잰 평균 총 10.9 m (10.1~11.6 m). 중앙선 좌우로 5.45 m씩.
  - 경로 3종: center(중앙선), mincurv(경계 안 곡률 제곱 합 최소), recorded(가장 빠른 기록 주행선).
  - 목표 속도: speed_profile(코너 한계 √(μg/|κ|) + 가속·감속 한계, μ 1.2, +4.3 / −5.1 m/s²)를 vref 로.
    recorded 경로에는 그 랩의 실제 기록 속도도 vref_rec 로 함께 넣는다(시간 순서대로 거리에 붙여 출발선 겹침 없음).
  - 출력 키: mppi.Track 이 읽는 형식 (x, y, psi, kappa, s, width_l, width_r, vref, bank, inner, outer) + grade, z, vref_rec.
"""
import argparse, glob, json, os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTER_CSV = os.path.join(ROOT, "data", "tracks", "ThunderhillWest_centerline.csv")
OUT = os.path.join(ROOT, "data", "tracks", "v2")
G = 9.81
WIDTH = 10.9                      # 팀원 실측 평균 (docs/track_width_measurement_thunderhill.pdf, 팀원 저장소)
CAR_HALF = 0.9                    # Golf 폭 1.79 m의 절반
MU0, AX_ACC0, AX_DEC0 = 1.2, 4.3, -5.1   # 팀원 기준값 (데이터에서 잰 값)


# ---------------------------------------------------------------- 기하 (팀원 track.py 와 같은 정의)
def resample_closed(p, ds=1.0):
    """닫힌 곡선을 균일 간격으로. 점 수 = round(길이/ds), 실제 간격 = 길이/점 수 (이음매까지 모두 같은 간격, 1 m 에서 ±0.02% 안)."""
    p = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    n = int(round(d[-1] / ds))
    s = np.arange(n) * (d[-1] / n)
    return np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], 1)


def normals(p):
    d = 0.5 * (np.roll(p, -1, 0) - np.roll(p, 1, 0))
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], 1)            # 왼쪽 법선


def heading_curv(p, smooth=15):
    d = np.gradient(p, axis=0)
    psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    dd = np.gradient(d, axis=0)
    kappa = (d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]) / np.hypot(*d.T) ** 3
    return psi, (uniform_filter1d(kappa, smooth, mode="wrap") if smooth else kappa)


def project(path, pts):
    idx = cKDTree(path).query(pts)[1]
    return idx, np.einsum("ij,ij->i", pts - path[idx], normals(path)[idx])


def speed_profile(kappa, ds=1.0, mu=MU0, ax_acc=AX_ACC0, ax_dec=AX_DEC0):
    """팀원 track.speed_profile 과 같음 (닫힌 트랙, 세 바퀴 돌려 이음매를 맞춤)."""
    vlim = np.sqrt(mu * G / np.maximum(np.abs(kappa), 1e-6))
    n = len(vlim)
    v = np.tile(vlim, 3)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * ax_acc * ds))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * abs(ax_dec) * ds))
    return v[n:2 * n]


def min_curvature_offsets(ref, lo, hi, e0=None):
    """팀원 track.min_curvature_offsets 와 같은 문제: Σ|p_{i+1} − 2p_i + p_{i−1}|² 최소, lo ≤ e ≤ hi.
    e0: 시작점 (기본 0 = 중앙선). 4차 미분 꼴이라 수렴이 느려 기록선에서 시작하고 반복을 늘린다."""
    nrm = normals(ref)

    def f(e):
        p = ref + e[:, None] * nrm
        dd = np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)
        gp = 2 * (np.roll(dd, -1, 0) - 2 * dd + np.roll(dd, 1, 0))
        return float((dd ** 2).sum()), np.einsum("ij,ij->i", gp, nrm)

    e0 = np.clip(np.zeros(len(ref)) if e0 is None else e0, lo, hi)
    sol = minimize(f, e0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                   options=dict(maxiter=50000, maxfun=100000, ftol=1e-14, gtol=1e-10))
    return sol.x, dict(success=bool(sol.success), message=str(sol.message), nit=int(sol.nit),
                       cost_center=float(f(np.zeros(len(ref)))[0]), cost_start=float(f(e0)[0]), cost=float(sol.fun))


# ---------------------------------------------------------------- D7
def d7_files(split_json=None, split="train"):
    from data.datasets import D7_DIR, D7_DUP
    fs = [f for f in sorted(glob.glob(os.path.join(D7_DIR, "*.csv"))) if os.path.basename(f)[:-4] not in D7_DUP]
    if split_json:
        keep = set(json.load(open(split_json, encoding="utf-8"))[split])
        fs = [f for f in fs if os.path.basename(f)[:-4] in keep]
    return fs


def fastest_lap(files=None):
    """평균 속도가 가장 빠른 닫힌 한 바퀴 (25 Hz, 출발점을 1초씩 옮기며, 이음매 1.5 m 안, v > 5)."""
    best = None
    for f in files or d7_files():
        a = pd.read_csv(f, comment="#", usecols=["posE_m", "posN_m", "Vx_mps"])[["posE_m", "posN_m", "Vx_mps"]].to_numpy()
        a = a[:len(a) // 8 * 8].reshape(-1, 8, 3).mean(1)
        x, y, v = a.T
        d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        for i0 in range(0, len(x), 25):
            far = np.where(d - d[i0] > 1500)[0]
            if not len(far):
                break
            j = far[0]
            k = np.where(np.hypot(x[j:] - x[i0], y[j:] - y[i0]) < 1.5)[0]
            if not len(k):
                continue
            k = j + k[0]
            if v[i0:k].min() > 5 and (best is None or v[i0:k].mean() > best["v_mean"]):
                best = dict(run=os.path.basename(f)[:-4], v_mean=float(v[i0:k].mean()), lap_time=(k - i0) * 0.04,
                            x=x[i0:k], y=y[i0:k], v=v[i0:k])
    return best


# ---------------------------------------------------------------- 경로
def make_path(center, half_l, half_r, road, offs=None, ds=1.0, mu=MU0, ax_acc=AX_ACC0, ax_dec=AX_DEC0, points=None):
    """중앙선 + 횡오프셋(offs) 또는 경로 점(points) → 경로. 경계·노면은 중앙선 좌표계에서 정의돼 있으므로 새 경로에서 다시 투영한다."""
    nrm = normals(center)
    p = resample_closed(points if points is not None else center + offs[:, None] * nrm, ds)
    psi, kappa = heading_curv(p)
    idx, ey = project(center, p)
    return dict(x=p[:, 0], y=p[:, 1], psi=psi, kappa=kappa, s=np.arange(len(p)) * ds,
                width_l=half_l[idx] - ey, width_r=half_r[idx] + ey, vref=speed_profile(kappa, ds, mu, ax_acc, ax_dec),
                bank=road["bank"][idx], grade=road["grade"][idx], z=road["z"][idx],
                inner=center + half_l[:, None] * nrm, outer=center - half_r[:, None] * nrm,
                mu=mu, ax_acc=ax_acc, ax_dec=ax_dec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=WIDTH, help="트랙 총 폭 [m] (기본 = 팀원 실측 평균 10.9)")
    ap.add_argument("--margin", type=float, default=1.0, help="mincurv 경로의 가장자리 여유 [m] (팀원 기본 1.0)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--width-points", default=None,
                    help="위치별 실측 폭 csv (no, s, width, ...). 주면 측정점 근처는 실측값, 먼 곳은 --width 로 이어 붙인다 "
                         "(예: data/tracks/thunderhill_width_points.csv, 출력은 --out 을 따로 주는 것을 권장)")
    ap.add_argument("--width-reach", type=float, default=30.0, help="측정점 하나가 영향을 주는 거리 척도 [m] (가우스 σ)")
    ap.add_argument("--center-csv", default=CENTER_CSV, help="중앙선·노면 csv (data.thunderhill_west 출력)")
    ap.add_argument("--split-json", default=None, help="분할 파일. 주면 기록 랩을 --split 파일에서만 고른다")
    ap.add_argument("--split", default="train")
    a = ap.parse_args(argv)

    T = np.loadtxt(a.center_csv, delimiter=",", comments="#")
    center = resample_closed(T[:, :2], 1.0)
    j = cKDTree(T[:, :2]).query(center)[1]
    road = dict(z=T[j, 4], grade=T[j, 5], bank=T[j, 6])
    n = len(center)
    width = np.full(n, a.width)
    if a.width_points:
        W = np.loadtxt(a.width_points, delimiter=",", comments="#", usecols=(1, 2), encoding="utf-8")
        W = W[np.argsort(W[:, 0])]
        S, Wv = W[:, 0] % n, W[:, 1]
        # 이웃 측정점 사이가 gap_max 이하면 선형 보간, 더 멀면 양 끝에서 width_reach[m]에 걸쳐 평균 폭으로 넘어간다
        gap_max = 150.0
        for i in range(n):
            k = np.searchsorted(S, i, side="right")
            s0, w0 = (S[k - 1], Wv[k - 1]) if k > 0 else (S[-1] - n, Wv[-1])
            s1, w1 = (S[k], Wv[k]) if k < len(S) else (S[0] + n, Wv[0])
            if s1 - s0 <= gap_max:
                width[i] = w0 + (w1 - w0) * (i - s0) / max(s1 - s0, 1e-9)
            else:
                t0 = np.exp(-0.5 * ((i - s0) / a.width_reach) ** 2)
                t1 = np.exp(-0.5 * ((s1 - i) / a.width_reach) ** 2)
                width[i] = (t0 * w0 + t1 * w1 + 0.05 * a.width) / (t0 + t1 + 0.05)
    half_l = half_r = width / 2

    # 1) D7 주행 위치가 새 경계 안에 들어오는가 (OSM 중앙선 위치 점검)
    allp = np.vstack([__import__("data.thunderhill_west", fromlist=["positions"]).positions(f)[::10] for f in d7_files()])
    idx, ey = project(center, allp)
    room = width[idx] / 2 - CAR_HALF
    over = np.abs(ey) > room
    over_s = np.unique(idx[over])
    check = dict(points=int(len(ey)), ey_abs_p50_p99_max=[float(np.percentile(np.abs(ey), q)) for q in (50, 99)] + [float(np.abs(ey).max())],
                 min_room_left_m=float((room - np.abs(ey)).min()),
                 frac_center_beyond_edge_minus_halfcar=float(over.mean()), frac_center_beyond_edge=float((np.abs(ey) > width[idx] / 2).mean()),
                 track_m_with_wheels_outside=int(len(over_s)))

    # 2) 경로 3종
    lap = fastest_lap(d7_files(a.split_json, a.split) if a.split_json else None)
    lap_xy = np.c_[lap["x"], lap["y"]]
    # mincurv 시작점용 오프셋 (기록선을 중앙선 좌표로)
    li, le = project(center, lap_xy)
    rec_off_raw = pd.Series(le).groupby(li).median()
    ii = np.arange(n)
    rec_off = np.interp(ii, rec_off_raw.index.to_numpy(), rec_off_raw.to_numpy(), period=n)
    lo, hi = -half_r + a.margin, half_l - a.margin
    mc, mcinfo = min_curvature_offsets(center, lo, hi, e0=rec_off)
    # recorded = 기록 랩의 실제 주행선 그대로 (RTK-GPS, 1 m 균일 재표본). 출발점은 중앙선 0 m 에 가장 가까운 점으로 맞춘다
    k0 = int(cKDTree(lap_xy).query(center[0])[1])
    paths = {"center": make_path(center, half_l, half_r, road, np.zeros(n)),
             "mincurv": make_path(center, half_l, half_r, road, mc),
             "recorded": make_path(center, half_l, half_r, road, points=np.roll(lap_xy, -k0, 0))}
    # recorded 경로의 실제 기록 속도: 그 랩을 시간 순서대로 누적 거리에 붙이고, 경로 거리로 보간 (출발선 겹침 없음)
    pr = paths["recorded"]
    dl = np.r_[0, np.cumsum(np.hypot(np.diff(lap["x"]), np.diff(lap["y"])))]
    k0 = int(cKDTree(np.c_[lap["x"], lap["y"]]).query([pr["x"][0], pr["y"][0]])[1])     # 경로 시작점 = 랩의 몇 번째 점
    sp = np.r_[0, np.cumsum(np.hypot(np.diff(pr["x"]), np.diff(pr["y"])))]
    s_lap = (dl - dl[k0]) % dl[-1] * (sp[-1] / dl[-1])
    o = np.argsort(s_lap)
    pr["vref_rec"] = np.maximum(uniform_filter1d(np.interp(sp, s_lap[o], lap["v"][o], period=sp[-1]), 15, mode="wrap"), 8.0)

    # 기록 점 → recorded 경로 "선"까지 거리 (경로 점까지 거리로 재면 1 m 간격 때문에 최대 0.5 m로 과대평가된다)
    q = np.r_[np.c_[pr["x"], pr["y"]], [[pr["x"][0], pr["y"][0]]]]
    dense = np.vstack([np.linspace(q[i], q[i + 1], 50, endpoint=False) for i in range(len(q) - 1)])
    d_rec = cKDTree(dense).query(lap_xy)[0]
    info = dict(base=f"OSM 노면 중앙선 ({os.path.relpath(a.center_csv, ROOT)})", width_total_m=a.width, length_m=n,
                split=dict(json=a.split_json, split=a.split if a.split_json else "all"),
                recorded_vs_lap_m=dict(mean=float(d_rec.mean()), max=float(d_rec.max())),
                width_points=a.width_points, width_min_median_max=[float(width.min()), float(np.median(width)), float(width.max())],
                d7_inside_check=check, fastest_lap=dict(run=lap["run"], v_mean=lap["v_mean"], lap_time=lap["lap_time"]),
                mincurv_opt=mcinfo, mu=MU0, ax_acc=AX_ACC0, ax_dec=AX_DEC0)
    for k, p in paths.items():
        info[k] = dict(length_m=float(p["s"][-1] + 1), kappa_absmax=float(np.abs(p["kappa"]).max()), sum_kappa2=float(np.sum(p["kappa"] ** 2)),
                       vref_min=float(p["vref"].min()), vref_max=float(p["vref"].max()), lap_time_profile=float(np.sum(1.0 / p["vref"])),
                       width_min=float(np.minimum(p["width_l"], p["width_r"]).min()),
                       offset_from_center_absmax=float(np.abs(project(center, np.c_[p["x"], p["y"]])[1]).max()))
    info["recorded"]["lap_time_vref_rec"] = float(np.sum(1.0 / pr["vref_rec"]))

    os.makedirs(a.out, exist_ok=True)
    for k, p in paths.items():
        np.savez(os.path.join(a.out, f"thunderhill_{k}.npz"), **{kk: np.asarray(v) for kk, v in p.items()})
    np.savez(os.path.join(a.out, "thunderhill_boundary.npz"), center=center, half_l=half_l, half_r=half_r, **road)
    json.dump(info, open(os.path.join(a.out, "thunderhill_info.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(info, indent=1, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    nrm = normals(center)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(*allp[::5].T, ",", color="#bbbbbb", alpha=0.5, label="D7 주행 위치")
    for sgn, hw in ((1, half_l), (-1, half_r)):
        ax.plot(*(center + sgn * hw[:, None] * nrm).T, color="k", lw=1)
    cols = dict(center="#888888", mincurv="#2FBF71", recorded="#E05A5A")
    for k, p in paths.items():
        ax.plot(p["x"], p["y"], color=cols[k], lw=1.2, label=f"{k} (프로파일 랩 {info[k]['lap_time_profile']:.1f} s)")
    if len(over_s):
        ax.plot(*center[over_s].T, "o", color="#F5883A", ms=3, label="D7 바퀴가 경계 밖으로 나가는 곳")
    ax.set_aspect("equal"); ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Thunderhill West v2: OSM 중앙선, 폭 {width.min():.1f}~{width.max():.1f} m"
                 + (" (실측 14곳 위치별)" if a.width_points else f" (실측 평균 {a.width} m 균일)"))
    fig.savefig(os.path.join(a.out, "thunderhill_v2.png"), dpi=130, bbox_inches="tight")
    print("saved", os.path.relpath(a.out, ROOT))


if __name__ == "__main__":
    main()
