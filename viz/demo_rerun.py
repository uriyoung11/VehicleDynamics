"""Rerun 3D 데모: viz/demo_track.py와 같은 시나리오(Spielberg 1:10, 젖은 코너)를 3D 장면 + 시계열로 기록한다.

  python -m viz.demo_rerun --ckpt checkpoints/f1tenth_v3_trial/m0_s0.pt checkpoints/f1tenth_v3_trial/b1_s0.pt
  rerun viz/out/demo.rrd          # 뷰어로 열기. 화면이 깨지면: rerun --renderer gl viz/out/demo.rrd

화면
  - 추격 카메라(큰 화면): 차 기준 좌표계라 카메라가 차 뒤를 따라 돈다.
      차(앞바퀴 조향 표시), 타이어 횡력 화살표(초록 여유 / 노랑 한계 진입 / 빨강 포화),
      트랙 중심선에서 차까지의 막대(가장자리까지 여유에 따라 색), 모델별 1초 뒤 고스트 차와 예측 궤적
  - 전체 서킷, 마찰원(지금 쓰는 마찰 / 쓸 수 있는 마찰), 상태판
  - 그래프: 가장자리까지 여유(지금, 1초 안 최소: 실제 vs 예측), 1초 뒤 위치 오차, r·Uy·속도·슬립각
시간 슬라이더로 앞뒤로 움직이며 볼 수 있다.
"""
import argparse, math, os, time
import numpy as np
import torch
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial import cKDTree

from physics import F1TENTH, forces
from models.baselines import load_checkpoint
from viz.demo_track import load_track, speed_profile, simulate, predict_paths, truth_paths, LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RGB = {"truth": [255, 255, 255], "m0": [255, 138, 61], "b1": [78, 163, 255], "m1": [61, 220, 132], "m2": [199, 125, 255]}
CAR, GRASS = [255, 196, 0], [24, 40, 28]
OK, WARN, BAD = [70, 200, 110], [250, 200, 40], [240, 70, 60]
Z_ROAD, Z_WET, Z_LINE, Z_PATH = 0.0, 0.004, 0.008, 0.03
P = F1TENTH
TRACK_HALF = 0.14   # 바퀴 좌우 간격의 절반 [m] (시각화용)


# ---------------------------------------------------------------- 계산

def strip_mesh(left, right, z, color):
    """두 경계선 사이를 삼각형으로 채운 메쉬(닫힌 고리)."""
    n = len(left)
    v = np.vstack([np.c_[left, np.full(n, z)], np.c_[right, np.full(n, z)]])
    tri = []
    for i in range(n):
        j = (i + 1) % n
        tri += [[i, j, n + i], [j, n + j, n + i]]
    return rr.Mesh3D(vertex_positions=v, triangle_indices=np.array(tri), vertex_colors=np.tile(color, (len(v), 1)))


class Margin:
    """트랙 가장자리까지 남은 거리. 양수면 트랙 안, 음수면 그만큼 밖으로 나감."""

    def __init__(self, tr):
        c = tr["center"]
        t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
        self.c, self.n = c, np.stack([-t[:, 1], t[:, 0]], 1)
        self.wl = np.linalg.norm(tr["left"] - c, axis=1)
        self.wr = np.linalg.norm(tr["right"] - c, axis=1)
        self.tree = cKDTree(c)

    def __call__(self, xy):
        xy = np.asarray(xy); flat = xy.reshape(-1, 2)
        _, j = self.tree.query(flat)
        e = np.sum((flat - self.c[j]) * self.n[j], 1)          # 중심선에서 왼쪽(+) 오프셋
        m = np.minimum(self.wl[j] - e, self.wr[j] + e)
        return m.reshape(xy.shape[:-1]), j.reshape(xy.shape[:-1])


def margin_color(m):
    return BAD if m < 0 else WARN if m < 0.25 else OK


def tire_state(sim):
    """기록된 상태·입력으로 타이어 횡력·수직 하중과 |α|/α_peak를 다시 계산."""
    T = lambda a: torch.as_tensor(a, dtype=torch.float64)
    x = torch.stack([T(sim[k]) for k in ("X", "Y", "psi", "ux", "uy", "r")], 1)
    u = torch.stack([T(sim["delta"]), T(sim["fx"])], 1)
    mu = T(sim["mu"])
    f = {k: v.numpy() for k, v in forces(x, u, P, mu).items()}
    out = {}
    for ax, tire in (("f", P.tire_f), ("r", P.tire_r)):
        a_peak = math.tan(math.pi / (2 * tire.C)) * tire.C * sim["mu"] / tire.c_alpha
        out["ratio_" + ax] = np.abs(f["alpha_" + ax]) / a_peak
        out["fy_" + ax] = f["fy" + ax] / (sim["mu"] * f["fz" + ax])     # 쓸 수 있는 마찰 대비
    return out


def regime_color(ratio):
    return BAD if ratio >= 1.0 else WARN if ratio >= 0.5 else OK


def scenario(args):
    """demo_track.main과 같은 시나리오. 시뮬레이션 결과는 viz/out/에 캐시."""
    tr = load_track(args.track)
    v_ref = speed_profile(tr, args.mu_dry, k_lat=args.k_lat)
    c = int(np.argmax(v_ref ** 2 * np.abs(tr["kappa"]))); s_c = tr["s"][c]
    wet = (max(0.0, s_c - 0.4 * args.wet_len), s_c + 0.6 * args.wet_len)
    lap_time = np.cumsum(np.r_[0, np.diff(tr["s"]) / np.maximum(v_ref[:-1], 0.3)])
    t_wet_lap = float(np.interp(wet[0], tr["s"], lap_time))
    t_start = max(0.0, t_wet_lap - args.before - 3.0)
    start = int(np.searchsorted(lap_time, t_start))
    duration = (t_wet_lap - t_start) + args.after + 2.0
    cache = os.path.join(ROOT, "viz", "out", f"sim_{args.track}_{args.mu_dry}_{args.mu_wet}_{args.k_lat}_{args.wet_len}.npz")
    if os.path.exists(cache):
        sim = dict(np.load(cache))
    else:
        t0 = time.time()
        sim = simulate(tr, v_ref, lambda s: args.mu_wet if wet[0] <= s <= wet[1] else args.mu_dry, duration, start=start)
        os.makedirs(os.path.dirname(cache), exist_ok=True); np.savez(cache, **sim)
        print(f"시뮬레이션 {time.time() - t0:.0f}s (캐시 저장)")
    return tr, wet, sim


# ---------------------------------------------------------------- 화면 구성

def blueprint():
    chase = rrb.Spatial3DView(
        origin="/world/car", contents=["/world/**", "- /world/marker/**"], name="추격 카메라",
        background=GRASS, line_grid=False,
        eye_controls=rrb.EyeControls3D(position=[-2.2, 0.0, 1.1], look_target=[1.2, 0.0, 0.0], eye_up=[0, 0, 1]))
    overview = rrb.Spatial3DView(origin="/world", contents=["/world/track/**", "/world/marker/**", "/world/trail/**"],
                                 name="전체 서킷", background=GRASS, line_grid=False)
    friction = rrb.Spatial2DView(origin="/friction", name="마찰원 (|가속도| / μg)", background=[20, 20, 24])
    plots = rrb.Vertical(
        rrb.TimeSeriesView(origin="/plots/margin", name="트랙 가장자리까지 여유 [m]  (0 아래 = 트랙 이탈)"),
        rrb.TimeSeriesView(origin="/plots/margin_pred", name="앞으로 1초 동안 최소 여유: 실제 vs 모델 예측 [m]"),
        rrb.TimeSeriesView(origin="/plots/error", name="1초 뒤 위치 오차 [m]"),
        rrb.Tabs(
            rrb.TimeSeriesView(origin="/plots/yaw_rate", name="요레이트 r (1초 예측)"),
            rrb.TimeSeriesView(origin="/plots/lat_vel", name="횡속도 Uy (1초 예측)"),
            rrb.TimeSeriesView(origin="/plots/speed", name="속도·μ"),
            rrb.TimeSeriesView(origin="/plots/slip", name="슬립각"),
        ),
        row_shares=[1, 1, 1, 1.2],
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(chase,
                         rrb.Horizontal(overview, friction, rrb.TextDocumentView(origin="/hud", name="상태"),
                                        column_shares=[1.3, 1, 1.1]),
                         row_shares=[2.6, 1.2]),
            plots,
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def log_static(tr, wet, paths, args):
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    L, R = tr["left"], tr["right"]
    rr.log("/world/track/asphalt", strip_mesh(L, R, Z_ROAD, [58, 60, 66]), static=True)
    # 연석: 빨강·흰색 줄무늬
    for side, line in (("l", L), ("r", R)):
        closed = np.r_[line, line[:1]]
        segs, cols = [], []
        for k, i in enumerate(range(0, len(closed) - 1, 3)):
            seg = closed[i:i + 4]
            segs.append(np.c_[seg, np.full(len(seg), Z_LINE)]); cols.append([225, 40, 40] if k % 2 else [240, 240, 240])
        rr.log(f"/world/track/kerb_{side}", rr.LineStrips3D(segs, colors=cols, radii=0.045), static=True)
    # 중심선 점선
    c = tr["center"]
    dashes = [np.c_[c[i:i + 3], np.full(len(c[i:i + 3]), Z_LINE)] for i in range(0, len(c) - 3, 6)]
    rr.log("/world/track/center", rr.LineStrips3D(dashes, colors=[[200, 200, 200, 110]], radii=0.012), static=True)
    idx = np.where((tr["s"] >= wet[0]) & (tr["s"] <= wet[1]))[0]
    if len(idx) > 1:
        m = len(idx)
        rr.log("/world/track/wet", rr.Mesh3D(
            vertex_positions=np.vstack([np.c_[L[idx], np.full(m, Z_WET)], np.c_[R[idx], np.full(m, Z_WET)]]),
            triangle_indices=np.array([[i, i + 1, m + i] for i in range(m - 1)] + [[i + 1, m + i + 1, m + i] for i in range(m - 1)]),
            vertex_colors=np.tile([47, 129, 247, 150], (2 * m, 1))), static=True)

    # 차: 차체 + 윗판 + 바퀴 4개(앞바퀴는 매 시점 조향각만큼 회전) — 1/10 RC카 실제 크기
    a, b = P.a, P.b
    rr.log("/world/car/body", rr.Boxes3D(centers=[[(a - b) / 2, 0, 0.06]], half_sizes=[[(a + b) / 2 + 0.09, 0.115, 0.03]],
                                         colors=[CAR], fill_mode="solid"), static=True)
    rr.log("/world/car/shell", rr.Boxes3D(centers=[[(a - b) / 2 - 0.02, 0, 0.11]], half_sizes=[[0.13, 0.09, 0.025]],
                                          colors=[[30, 30, 34]], fill_mode="solid"), static=True)
    rr.log("/world/car/nose", rr.Boxes3D(centers=[[a + 0.11, 0, 0.05]], half_sizes=[[0.02, 0.12, 0.012]],
                                         colors=[[220, 40, 40]], fill_mode="solid"), static=True)
    wheel = rr.Cylinders3D(lengths=[0.045], radii=[0.05], centers=[[0, 0, 0]], colors=[[25, 25, 25]], fill_mode="solid",
                           rotation_axis_angles=[rr.RotationAxisAngle(axis=[1, 0, 0], degrees=90)])
    for name, pos in (("fl", [a, TRACK_HALF]), ("fr", [a, -TRACK_HALF]), ("rl", [-b, TRACK_HALF]), ("rr", [-b, -TRACK_HALF])):
        rr.log(f"/world/car/wheel_{name}", rr.Transform3D(translation=[*pos, 0.05]), static=True)
        rr.log(f"/world/car/wheel_{name}/tire", wheel, static=True)

    # 시계열 스타일
    rr.log("/plots/margin/now", rr.SeriesLines(colors=[CAR], names=["지금 여유"], widths=[2.5]), static=True)
    rr.log("/plots/margin/edge", rr.SeriesLines(colors=[[240, 70, 60]], names=["트랙 경계"], widths=[1]), static=True)
    rr.log("/plots/margin_pred/truth", rr.SeriesLines(colors=[[235, 235, 235]], names=["실제"], widths=[2.5]), static=True)
    rr.log("/plots/margin_pred/edge", rr.SeriesLines(colors=[[240, 70, 60]], names=["트랙 경계"], widths=[1]), static=True)
    for k in paths:
        rr.log(f"/plots/margin_pred/{k}", rr.SeriesLines(colors=[RGB[k]], names=[f"{LABELS[k]} 예측"], widths=[1.8]), static=True)
        rr.log(f"/plots/error/{k}", rr.SeriesLines(colors=[RGB[k]], names=[LABELS[k]], widths=[2]), static=True)
        rr.log(f"/plots/yaw_rate/{k}", rr.SeriesLines(colors=[RGB[k]], names=[f"{LABELS[k]} 예측"], widths=[1.5]), static=True)
        rr.log(f"/plots/lat_vel/{k}", rr.SeriesLines(colors=[RGB[k]], names=[f"{LABELS[k]} 예측"], widths=[1.5]), static=True)
    for p_ in ("yaw_rate", "lat_vel"):
        rr.log(f"/plots/{p_}/truth", rr.SeriesLines(colors=[[235, 235, 235]], names=["실제"], widths=[2.5]), static=True)
    rr.log("/plots/speed/ux", rr.SeriesLines(colors=[CAR], names=["속도 [m/s]"], widths=[2]), static=True)
    rr.log("/plots/speed/mu", rr.SeriesLines(colors=[[47, 129, 247]], names=["노면 μ ×10"], widths=[2]), static=True)
    rr.log("/plots/slip/beta", rr.SeriesLines(colors=[CAR], names=["차체 슬립 β [°]"], widths=[2]), static=True)
    rr.log("/plots/slip/af", rr.SeriesLines(colors=[[255, 120, 120]], names=["앞바퀴 α_f [°]"], widths=[1.5]), static=True)
    rr.log("/plots/slip/ar", rr.SeriesLines(colors=[[160, 220, 255]], names=["뒷바퀴 α_r [°]"], widths=[1.5]), static=True)

    # 마찰원 (2D 화면은 y가 아래로 커지므로 가속을 위로 그리려고 부호를 뒤집는다)
    th = np.linspace(0, 2 * np.pi, 121)
    rr.log("/friction/limit", rr.LineStrips2D([np.c_[np.cos(th), np.sin(th)]], colors=[[240, 70, 60]], radii=0.012), static=True)
    rr.log("/friction/half", rr.LineStrips2D([np.c_[0.5 * np.cos(th), 0.5 * np.sin(th)]], colors=[[110, 110, 120]], radii=0.006),
           static=True)
    rr.log("/friction/axes", rr.LineStrips2D([[[-1.15, 0], [1.15, 0]], [[0, -1.15], [0, 1.15]]], colors=[[80, 80, 90]], radii=0.004),
           static=True)
    rr.log("/friction/labels", rr.Points2D([[0, -1.3], [0, 1.3], [-1.3, 0], [1.3, 0]], radii=0.001,
                                           labels=["가속", "제동", "좌회전", "우회전"], colors=[[180, 180, 190]]), static=True)


def hud_text(sim, s, margin, ts, args, paths, fde_now):
    mu = sim["mu"][s]; wet = mu < args.mu_dry
    m = margin[s]
    state = "🔴 **트랙 이탈**" if m < 0 else "🟡 가장자리 근접" if m < 0.25 else "🟢 여유 있음"
    rf, rrr = ts["ratio_f"][s], ts["ratio_r"][s]
    lab = lambda x: "🔴 포화" if x >= 1 else "🟡 한계 진입" if x >= 0.5 else "🟢 평소"
    lines = [
        f"### t = {sim['t'][s]:5.2f} s",
        f"- 속도 **{sim['ux'][s]:.2f} m/s**  ·  노면 μ **{mu:.2f}** {'(젖음)' if wet else '(마름)'}",
        f"- 가장자리까지 **{m:+.2f} m**  {state}",
        f"- 앞바퀴 {lab(rf)} ({rf:.2f})  ·  뒷바퀴 {lab(rrr)} ({rrr:.2f})",
        "", "**1초 뒤 위치 오차**",
    ]
    lines += [f"- {LABELS[k]}: {v:.2f} m" for k, v in fde_now.items()]
    lines += ["", "<small>흰 고스트 = 실제 1초 뒤 위치, 색 고스트 = 모델이 지금 예측한 1초 뒤 위치. "
              "바퀴 위 화살표 = 타이어 횡력(길이: 쓸 수 있는 마찰 대비)</small>"]
    return "\n".join(lines)


# ---------------------------------------------------------------- 메인

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--mu-dry", type=float, default=0.7)
    ap.add_argument("--mu-wet", type=float, default=0.55)
    ap.add_argument("--k-lat", type=float, default=0.75)
    ap.add_argument("--wet-len", type=float, default=25.0)
    ap.add_argument("--path-hz", type=int, default=20, help="예측 궤적을 초당 몇 번 그릴지")
    ap.add_argument("--before", type=float, default=8.0)
    ap.add_argument("--after", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "viz", "out", "demo.rrd"))
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    tr, wet, sim = scenario(args)
    n = len(sim["t"])
    frames = np.arange(4, n - 101, 100 // args.path_hz)
    paths, models = {}, {}
    for p in args.ckpt:
        model, ck = load_checkpoint(p)
        models[ck["kind"]] = model
        paths[ck["kind"]] = predict_paths(model, sim, frames)
    truth = truth_paths(sim, frames)
    fde = {k: np.linalg.norm(paths[k][:, -1] - truth[:, -1], axis=1) for k in paths}
    mfun = Margin(tr)
    margin, near = mfun(np.c_[sim["X"], sim["Y"]])
    mp_truth = mfun(truth)[0].min(1)
    mp_pred = {k: mfun(v)[0].min(1) for k, v in paths.items()}
    ts = tire_state(sim)

    rr.init("vehicle_dynamics_demo")
    rr.save(args.out, default_blueprint=blueprint())
    log_static(tr, wet, paths, args)

    t0 = time.time()
    fpos = {f: i for i, f in enumerate(frames)}
    G = 9.81
    fde_now = {k: float("nan") for k in paths}
    for s in range(n):
        rr.set_time("time", duration=float(sim["t"][s]))
        X, Y, psi, de = (float(sim[k][s]) for k in ("X", "Y", "psi", "delta"))
        rr.log("/world/car", rr.Transform3D(translation=[X, Y, 0.0], rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=psi)))
        for w in ("fl", "fr"):
            rr.log(f"/world/car/wheel_{w}/tire", rr.Transform3D(rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=de)))
        # 타이어 횡력 화살표 (차 좌표계). 앞은 바퀴 방향에 수직
        vf = 0.45 * ts["fy_f"][s]; vr = 0.45 * ts["fy_r"][s]
        rr.log("/world/car/forces", rr.Arrows3D(
            origins=[[P.a, 0, 0.16], [-P.b, 0, 0.16]],
            vectors=[[-math.sin(de) * vf, math.cos(de) * vf, 0], [0, vr, 0]],
            colors=[regime_color(ts["ratio_f"][s]), regime_color(ts["ratio_r"][s])], radii=0.018))
        # 중심선 → 차 막대 (가장자리까지 여유에 따라 색)
        cj = tr["center"][near[s]]
        rr.log("/world/offset", rr.LineStrips3D([[[cj[0], cj[1], Z_PATH], [X, Y, Z_PATH]]], colors=[margin_color(margin[s])],
                                                radii=0.02))
        rr.log("/plots/margin/now", rr.Scalars(margin[s]))
        rr.log("/plots/margin/edge", rr.Scalars(0.0))
        rr.log("/plots/speed/ux", rr.Scalars(sim["ux"][s]))
        rr.log("/plots/speed/mu", rr.Scalars(10 * sim["mu"][s]))
        rr.log("/plots/slip/beta", rr.Scalars(np.degrees(np.arctan2(sim["uy"][s], sim["ux"][s]))))
        rr.log("/plots/slip/af", rr.Scalars(np.degrees(sim["alpha_f"][s])))
        rr.log("/plots/slip/ar", rr.Scalars(np.degrees(sim["alpha_r"][s])))
        if s + 100 < n:
            rr.log("/plots/yaw_rate/truth", rr.Scalars(sim["r"][s + 100]))
            rr.log("/plots/lat_vel/truth", rr.Scalars(sim["uy"][s + 100]))
        if s % 5 == 0:
            # 지나온 길: 여유에 따라 색이 바뀌는 3초 꼬리
            a0 = max(0, s - 300)
            pts = np.c_[sim["X"][a0:s + 1], sim["Y"][a0:s + 1], np.full(s + 1 - a0, Z_PATH)]
            segs = [pts[i:i + 6] for i in range(0, len(pts) - 1, 5)]
            cols = [margin_color(margin[a0 + i]) for i in range(0, len(pts) - 1, 5)]
            rr.log("/world/trail", rr.LineStrips3D(segs, colors=cols, radii=0.02))
            rr.log("/world/marker", rr.Points3D([[X, Y, 0.3]], colors=[CAR], radii=0.9))
            mu = sim["mu"][a0:s + 1]
            ay, ax = sim["ay"][a0:s + 1] / (mu * G), sim["ax"][a0:s + 1] / (mu * G)
            rr.log("/friction/trail", rr.LineStrips2D([np.c_[-ay[-100:], -ax[-100:]]], colors=[[255, 196, 0, 120]], radii=0.01))
            rr.log("/friction/now", rr.Points2D([[-ay[-1], -ax[-1]]], colors=[CAR], radii=0.05))
        if s in fpos:
            fi = fpos[s]
            for k in ["truth", *paths]:
                path = truth[fi] if k == "truth" else paths[k][fi]
                rr.log(f"/world/pred/{k}", rr.LineStrips3D([np.c_[path, np.full(len(path), Z_PATH + (0.01 if k == 'truth' else 0))]],
                                                           colors=[RGB[k]], radii=0.03 if k == "truth" else 0.025))
                # 1초 뒤 고스트 차 (마지막 구간의 방향으로 놓음)
                hd = math.atan2(*(path[-1] - path[-4])[::-1])
                col = RGB[k] + [70 if k == "truth" else 110]
                rr.log(f"/world/ghost/{k}", rr.Transform3D(translation=[*path[-1], 0.0],
                                                           rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=hd)))
                rr.log(f"/world/ghost/{k}/box", rr.Boxes3D(centers=[[(P.a - P.b) / 2, 0, 0.06]],
                                                           half_sizes=[[(P.a + P.b) / 2 + 0.09, 0.115, 0.05]], colors=[col],
                                                           fill_mode="solid" if k != "truth" else "majorwireframe",
                                                           radii=0.012))
            rr.log("/plots/margin_pred/truth", rr.Scalars(mp_truth[fi]))
            rr.log("/plots/margin_pred/edge", rr.Scalars(0.0))
            for k in paths:
                rr.log(f"/plots/margin_pred/{k}", rr.Scalars(mp_pred[k][fi]))
                rr.log(f"/plots/error/{k}", rr.Scalars(fde[k][fi]))
                fde_now[k] = fde[k][fi]
        if s % 10 == 0:
            rr.log("/hud", rr.TextDocument(hud_text(sim, s, margin, ts, args, paths, fde_now), media_type=rr.MediaType.MARKDOWN))
    # 모델별 1초 뒤 r, Uy 예측값 (그 예측을 한 시각에 찍음)
    for k, model in models.items():
        rU = rollout_last(model, sim, frames)
        for fi, s in enumerate(frames):
            rr.set_time("time", duration=float(sim["t"][s]))
            rr.log(f"/plots/yaw_rate/{k}", rr.Scalars(rU[fi, 0]))
            rr.log(f"/plots/lat_vel/{k}", rr.Scalars(rU[fi, 1]))
    rr.disconnect()
    print(f"기록 {time.time() - t0:.0f}s -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print(f"가장자리까지 최소 여유 {margin.min():+.2f} m ({'트랙 이탈' if margin.min() < 0 else '트랙 안'}), "
          f"앞바퀴 포화 {np.mean(ts['ratio_f'] >= 1):.0%}, 뒷바퀴 포화 {np.mean(ts['ratio_r'] >= 1):.0%}")
    for k in paths:
        print(f"{k}: 1초 뒤 위치 오차 평균 {fde[k].mean():.2f} m, 최대 {fde[k].max():.2f} m")


@torch.no_grad()
def rollout_last(model, sim, starts, H=100, T=4, long_scale=F1TENTH.m * 100):
    """starts 시점에서 1초(H스텝) 롤아웃한 마지막 [r, Uy]."""
    F = torch.as_tensor(np.stack([sim["r"], sim["uy"], sim["ux"], sim["delta"], sim["fxf"] / long_scale], 1))
    hist = torch.stack([F[s - T + 1:s + 1] for s in starts])
    for k in range(H):
        pred = model(hist)
        nxt = torch.as_tensor(np.clip(np.array(starts) + k + 1, 0, len(F) - 1))
        hist = torch.cat([hist[:, 1:], torch.cat([pred, F[nxt, 2:]], -1)[:, None]], 1)
    return pred.numpy()


if __name__ == "__main__":
    main()
