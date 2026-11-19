"""Thunderhill West 트랙 중심선과 폭을 만든다 (공개 경계 파일이 없어서).

  python -m data.thunderhill_west                  # OSM 중앙선 + 레이싱 라인으로 폭 (기본)
  python -m data.thunderhill_west --source driven  # 이전 방식: 주행선 양옆 고정 폭
  -> data/tracks/ThunderhillWest_centerline.csv, docs/figures/thunderhill_west.png

osm (기본)
  1. D7에서 깨끗한 한 바퀴를 찾아 경로 기준으로 쓴다.
  2. OpenStreetMap(© OpenStreetMap contributors, ODbL)의 West 코스 highway=raceway 선(노면 중앙선)에서
     그 경로를 따라 가장 가까운 점을 모아 중앙선으로 쓴다. D7 좌표는 D1 트랙 자료와 같은 지역 좌표계다.
  3. 폭: 지점마다 레이싱 라인이 좌우로 간 최대 거리 + 차 반폭. 레이싱 라인이 가지 않는 쪽은 최소 반폭.
driven (이전)
  주행 위치의 좌우 범위 가운데를 중심선으로, 범위 + 여유를 폭으로 쓴다. 자율주행 차가 거의 같은 선만 달려
  범위가 1 m 남짓이라 사실상 "주행선 양옆 고정 폭"이 된다.
주의: 어느 쪽이든 공식 노면 경계가 아니다. 폭은 실제보다 좁을 수 있다(MPC에는 보수적).
형식은 f1tenth_racetracks와 같다: x_m, y_m, w_tr_right_m, w_tr_left_m (viz.demo_track.load_track으로 읽힘).
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def positions(f, extra=False):
    """Ux > 5 m/s 위치 [동, 북]. extra=True면 [동, 북, 고도, 롤, 횡가속도 Ux·r]."""
    cols = ["posE_m", "posN_m", "Vx_mps"] + (["posU_m", "rollAngle_rad", "yawRate_radps"] if extra else [])
    d = pd.read_csv(f, comment="#", usecols=cols)
    d = d[d.Vx_mps > 5]
    if not extra:
        return d[["posE_m", "posN_m"]].to_numpy()
    return np.c_[d.posE_m, d.posN_m, d.posU_m, d.rollAngle_rad, d.Vx_mps * d.yawRate_radps]


def smooth_onto_line(ref, pts, sigma):
    """꺾인 지도 중앙선 다듬기: 모양은 부드러운 레이싱 라인(ref)에서, 위치는 지도 선(pts, ref와 같은 순서)에서.
    ref에서 지도 선까지 좌우 거리를 sigma[m]로 평활해 ref에 더한다. 짧게 튀는 꺾임은 평활에서 사라진다."""
    t = np.gradient(ref, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    e = np.sum((pts - ref) * n, 1)
    return resample_closed(ref + n * gaussian_filter1d(e, sigma, mode="wrap")[:, None], 1.0, sigma=1.0)


def heading_jump(c):
    """중앙선 꺾임 정도: 1 m마다 방향 변화의 최댓값 [도]."""
    t = np.diff(np.r_[c, c[:1]], axis=0)
    h = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    return float(np.degrees(np.abs(np.diff(h)).max()))


def surface_profile(center, samples, ds=1.0, iters=8):
    """트랙 위치별 고도, 경사, 좌우 기울기.
    고도: 위치별 고도 중앙값(여러 바퀴). 경사: 고도를 트랙 거리로 미분(차체 피치를 쓰지 않아 가감속 영향이 없다).
    좌우 기울기: 롤 = 노면 기울기(위치마다) + k·횡가속도 로 보고 번갈아 푼다. 같은 위치를 다른 속도로 지난
    여러 바퀴의 차이로 k(차체가 횡가속도에 따라 기우는 정도)를 구한다."""
    j = cKDTree(center).query(samples[:, :2])[1]
    n = len(center)
    idx = np.arange(n)

    def per_point(v):   # 위치별 중앙값, 빈 곳은 보간 후 5 m 평활
        m = pd.Series(v).groupby(j).median().reindex(idx).to_numpy()
        ok = np.isfinite(m)
        return gaussian_filter1d(np.interp(idx, idx[ok], m[ok]), 5, mode="wrap")

    z = per_point(samples[:, 2])
    grade = np.gradient(np.r_[z[-2:], z, z[:2]], ds)[2:-2]          # 오르막 +
    roll, ay = samples[:, 3], samples[:, 4]
    bank, k = np.zeros(n), 0.0
    for _ in range(iters):
        k = float(np.linalg.lstsq(ay[:, None], roll - bank[j], rcond=None)[0][0])
        bank = per_point(roll - k * ay)
    return z, grade, bank, k


def find_lap(xy, min_len=1000.0, close=1.0):   # 이음매 허용 1 m (5 m면 시작·끝이 1.4 m 어긋난 채 이어져 꺾였다)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    for i0 in range(0, len(xy) // 3, 200):
        far = np.flatnonzero(s - s[i0] > min_len)
        if not len(far):
            break
        j = far[0] + np.argmin(np.linalg.norm(xy[far[0]:] - xy[i0], axis=1)[: 20000])
        if np.linalg.norm(xy[j] - xy[i0]) < close:
            return xy[i0:j]
    return None


def resample_closed(xy, ds=1.0, sigma=2.0):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[xy, xy[:1]], axis=0), axis=1))]
    n = int(s[-1] / ds)
    t = np.linspace(0, s[-1], n, endpoint=False)
    p = np.c_[np.interp(t, s, np.r_[xy[:, 0], xy[0, 0]]), np.interp(t, s, np.r_[xy[:, 1], xy[0, 1]])]
    return np.c_[gaussian_filter1d(p[:, 0], sigma, mode="wrap"), gaussian_filter1d(p[:, 1], sigma, mode="wrap")]


# D1(Spielberg) 트랙 자료의 지역 좌표 기준점(ECEF). D7 위치(posE, posN)도 같은 기준이다(OSM 선과 2 m 안으로 겹침, 11/17 확인)
REF_ECEF = np.array([-2634230.47253972, -4161936.65368287, 4038639.74184999])   # 북위 39.538493°, 서경 122.331103°
OSM_CACHE = os.path.join(ROOT, "data", "tracks", "thunderhill_osm.json")
OSM_WEST_WAYS = (756822999, 756823000, 1016088342)    # D7 경로와 겹치는 OSM highway=raceway 선 (Thunderhill West)


def fetch_osm():
    """OpenStreetMap(ODbL) Thunderhill 주변 highway=raceway 선. 처음 한 번 Overpass API로 받아 data/tracks/에 캐시한다."""
    import json
    if not os.path.exists(OSM_CACHE):
        import urllib.parse, urllib.request
        q = ('[out:json][timeout:60];(way["highway"="raceway"](around:3500,39.5385,-122.3311););out tags geom;')
        req = urllib.request.Request("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": "VehicleDynamicsResearch/0.1 (academic project)", "Accept": "application/json"})
        os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)
        open(OSM_CACHE, "wb").write(urllib.request.urlopen(req, timeout=90).read())
    return json.load(open(OSM_CACHE, encoding="utf-8"))


def to_local(lat, lon):
    """위도·경도 → D1/D7 지역 좌표 [동쪽, 북쪽] (m)."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    la, lo = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    X = np.c_[N * np.cos(la) * np.cos(lo), N * np.cos(la) * np.sin(lo), N * (1 - e2) * np.sin(la)] - REF_ECEF
    la0, lo0 = np.radians(39.538493), np.radians(-122.331103)
    R = np.array([[-np.sin(lo0), np.cos(lo0), 0], [-np.sin(la0) * np.cos(lo0), -np.sin(la0) * np.sin(lo0), np.cos(la0)]])
    return X @ R.T


def osm_centerline(ref, max_dist=8.0, raw=False):
    """D7 한 바퀴(ref)를 따라가며 가장 가까운 OSM 노면 중앙선 점을 모은다(D7이 달린 경로 부분만).
    raw=True면 ref와 같은 순서의 지도 점(다듬기 전)도 함께 돌려준다."""
    d = fetch_osm()
    dense = []
    for e in d["elements"]:
        if e["id"] not in OSM_WEST_WAYS:
            continue
        Q = to_local(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))
        for i in range(len(Q) - 1):
            n = max(2, int(np.linalg.norm(Q[i + 1] - Q[i]) / 0.25))
            dense.append(np.linspace(Q[i], Q[i + 1], n))
    dense = np.vstack(dense)
    dist, k = cKDTree(dense).query(ref)
    assert dist.max() < max_dist, f"OSM 선이 D7 경로에서 {dist.max():.1f} m 떨어진 곳이 있다"
    c = resample_closed(dense[k], 1.0, sigma=3.0)
    return (c, dense[k]) if raw else c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["osm", "driven"], default="osm",
                    help="osm: OpenStreetMap 노면 중앙선 + 레이싱 라인 범위로 폭 (기본). driven: 주행선 양옆 고정 폭 (이전 방식)")
    ap.add_argument("--margin", type=float, default=1.0, help="driven: 다닌 범위 바깥에 더할 여유 [m]")
    ap.add_argument("--min-width", type=float, default=8.8, help="최소 트랙 폭 [m] (Stanford 공개 Thunderhill 본 코스 경계 폭 중앙값 8.8 m)")
    ap.add_argument("--car-half", type=float, default=0.9, help="osm: 차 반폭 [m]. 레이싱 라인(차 중심) 바깥으로 이만큼 노면이 있어야 한다")
    ap.add_argument("--window", type=int, default=15, help="osm: 폭을 볼 앞뒤 범위 [m]")
    ap.add_argument("--smooth", type=float, default=10.0, help="osm: 지도 중앙선 꺾임 다듬기 평활 폭 [m]. 0이면 다듬지 않음")
    ap.add_argument("--split-json", default=None, help="분할 파일(json, train/val/test 목록). 주면 --split 파일만 쓴다 (시험 주행 정보가 지도에 섞이지 않게)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None, help="출력 csv (기본 data/tracks/ThunderhillWest_centerline.csv). 그림은 같은 폴더에 저장")
    args = ap.parse_args(argv)
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, "data", "d7", "*.csv")))
             if os.path.basename(f)[:-4] not in ("2023-10_2", "2023-10_3")]
    if args.split_json:
        import json
        keep = set(json.load(open(args.split_json, encoding="utf-8"))[args.split])
        files = [f for f in files if os.path.basename(f)[:-4] in keep]
        print(f"{args.split} 파일 {len(files)}개만 사용: {sorted(keep)}", flush=True)
    P = [positions(f, extra=True) for f in files]
    samples = np.vstack(P)                                         # [동, 북, 고도, 롤, 횡가속도]
    P = [p[:, :2] for p in P]
    lap = next((l for l in (find_lap(p) for p in P) if l is not None), None)
    assert lap is not None, "닫힌 한 바퀴를 찾지 못함"
    ref = resample_closed(lap)
    t = np.gradient(ref, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)                       # 진행 방향 왼쪽
    allp = np.vstack(P)
    if args.source == "osm":
        center, raw_pts = osm_centerline(ref, raw=True)
        jump0 = heading_jump(center)
        if args.smooth > 0:
            center = smooth_onto_line(ref, raw_pts, args.smooth)
        print(f"중앙선 꺾임(1 m당 최대 방향 변화): 지도 그대로 {jump0:.2f}° → 다듬은 뒤 {heading_jump(center):.2f}° "
              f"(레이싱 라인 {heading_jump(ref):.2f}°)", flush=True)
        t = np.gradient(center, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        j = cKDTree(center).query(allp)[1]
        e = np.sum((allp - center[j]) * nrm[j], 1)                 # 레이싱 라인의 중앙선 대비 오프셋 (왼쪽 +)
        hl, hr = np.full(len(center), np.nan), np.full(len(center), np.nan)
        for k in range(len(center)):
            w = e[np.abs(j - k) <= args.window]
            if len(w) > 20:
                hl[k], hr[k] = max(np.percentile(w, 99.5), 0.0), max(-np.percentile(w, 0.5), 0.0)
        idx = np.arange(len(center)); ok = np.isfinite(hl)
        hl = np.interp(idx, idx[ok], hl[ok]); hr = np.interp(idx, idx[ok], hr[ok])
        # 반폭 = 레이싱 라인이 그쪽으로 간 최대 거리 + 차 반폭. 데이터가 없는 쪽은 최소 반폭
        wl = gaussian_filter1d(np.maximum(hl + args.car_half, args.min_width / 2), 5, mode="wrap")
        wr = gaussian_filter1d(np.maximum(hr + args.car_half, args.min_width / 2), 5, mode="wrap")
        note = "OSM 노면 중앙선 + D7 레이싱 라인 범위로 추정한 폭"
        spread = hl + hr
    else:
        j = cKDTree(ref).query(allp)[1]
        e = np.sum((allp - ref[j]) * nrm[j], 1)
        lo, hi = np.full(len(ref), np.nan), np.full(len(ref), np.nan)
        for k in range(len(ref)):
            w = e[np.abs(j - k) <= 2]                                   # 앞뒤 2 m 안
            if len(w) > 20:
                lo[k], hi[k] = np.percentile(w, [0.5, 99.5])
        ok = np.isfinite(lo)
        idx = np.arange(len(ref))
        lo = np.interp(idx, idx[ok], lo[ok]); hi = np.interp(idx, idx[ok], hi[ok])
        lo, hi = gaussian_filter1d(lo, 5, mode="wrap"), gaussian_filter1d(hi, 5, mode="wrap")
        mid = (lo + hi) / 2
        center = ref + nrm * mid[:, None]
        wl = wr = np.maximum((hi - lo) / 2 + args.margin, args.min_width / 2)
        note = "D7 주행 범위 기반 근사 (중심선 = 주행선)"
        spread = hi - lo
    half = (wl + wr) / 2
    z, grade, bank, k_roll = surface_profile(center, samples)
    print(f"노면: 고도 범위 {z.max() - z.min():.1f} m, 경사 1~99분위 {np.degrees(np.percentile(grade, [1, 99])).round(1)}°, "
          f"좌우 기울기 1~99분위 {np.degrees(np.percentile(bank, [1, 99])).round(1)}°, "
          f"차체 롤 계수 {np.degrees(k_roll):.3f}°/(m/s²)", flush=True)
    out = args.out or os.path.join(ROOT, "data", "tracks", "ThunderhillWest_centerline.csv")
    fig_dir = os.path.dirname(os.path.abspath(out)) if args.out else os.path.join(ROOT, "docs", "figures")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # 앞 4열은 f1tenth_racetracks 형식(load_track이 읽음). 뒤 3열: 고도 [m], 경사 [rad, 오르막 +], 좌우 기울기 [rad, D7 롤과 같은 부호]
    np.savetxt(out, np.c_[center, wr, wl, z, np.arctan(grade), bank], delimiter=",", fmt="%.5f",
               header=f"x_m, y_m, w_tr_right_m, w_tr_left_m, z_m, grade_rad, bank_rad  ({note}, data/thunderhill_west.py)")
    L = np.sum(np.linalg.norm(np.diff(np.r_[center, center[:1]], axis=0), axis=1))
    print(f"[{args.source}] 한 바퀴 {L:.0f} m, 점 {len(center)}개, 레이싱 라인 좌우 범위 p5/50/95 {np.percentile(spread, [5, 50, 95]).round(1)} m, "
          f"최종 폭 p5/50/95 {np.percentile(wl + wr, [5, 50, 95]).round(1)} m, 투영 점 {len(allp)}개 (파일 {len(files)}개)")
    print("saved", os.path.relpath(out, ROOT))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(*allp[::20].T, ",", color="#999999", alpha=0.4, label="D7 주행 위치 (20개 파일)")
    ax.plot(*(center + nrm * wl[:, None]).T, color="k", lw=1)
    ax.plot(*(center - nrm * wr[:, None]).T, color="k", lw=1)
    ax.plot(*center.T, color="#1f77b4", lw=1, ls="--", label="중심선")
    ax.set_aspect("equal"); ax.legend(loc="best"); ax.set_title(f"Thunderhill West ({note}, 한 바퀴 {L:.0f} m)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "thunderhill_west.png"), dpi=120)

    s_ = np.arange(len(center)) * 1.0
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(s_, z - z.min(), color="#1f77b4"); axs[0].set_ylabel("고도 [m]")
    axs[1].plot(s_, np.degrees(np.arctan(grade)), color="#d62728"); axs[1].set_ylabel("경사 [°] (오르막 +)")
    axs[2].plot(s_, np.degrees(bank), color="#2ca02c"); axs[2].set_ylabel("좌우 기울기 [°]")
    for a in axs:
        a.grid(alpha=0.3); a.axhline(0, color="k", lw=0.5)
    axs[2].set_xlabel("트랙 거리 [m]")
    axs[0].set_title("Thunderhill West 노면 (D7 20개 파일, 위치별 중앙값)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "thunderhill_west_profile.png"), dpi=120)


if __name__ == "__main__":
    main()
