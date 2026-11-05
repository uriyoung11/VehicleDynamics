"""Rerun 3D 데모: viz/demo_track.py와 같은 시나리오(Spielberg 1:10, 젖은 코너)를 3D 장면 + 시계열로 기록한다.

  python -m viz.demo_rerun --ckpt checkpoints/f1tenth_v2_trial/m0_s0.pt checkpoints/f1tenth_v2_trial/b1_s0.pt
  rerun viz/out/demo.rrd          # 뷰어로 열기 (pip install rerun-sdk 에 뷰어 포함)

화면: 왼쪽 3D 서킷(차를 따라가는 카메라), 오른쪽 그래프(속도, 슬립각, 요레이트·횡속도 예측 vs 실제, 1초 뒤 위치 오차, 노면 μ).
시간 슬라이더로 앞뒤로 움직이며 볼 수 있다.
"""
import argparse, os, time
import numpy as np
import torch
import rerun as rr
import rerun.blueprint as rrb

from physics import F1TENTH
from models.baselines import load_checkpoint
from viz.demo_track import load_track, speed_profile, simulate, predict_paths, truth_paths, LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RGB = {"truth": [255, 255, 255], "m0": [255, 138, 61], "b1": [78, 163, 255], "m1": [61, 220, 132], "m2": [199, 125, 255]}
CAR = [255, 211, 61]
Z_ROAD, Z_WET, Z_LINE, Z_PATH = 0.0, 0.004, 0.008, 0.03


def strip_mesh(left, right, z, color):
    """두 경계선 사이를 삼각형으로 채운 메쉬(닫힌 고리)."""
    n = len(left)
    v = np.vstack([np.c_[left, np.full(n, z)], np.c_[right, np.full(n, z)]])
    tri = []
    for i in range(n):
        j = (i + 1) % n
        tri += [[i, j, n + i], [j, n + j, n + i]]
    return rr.Mesh3D(vertex_positions=v, triangle_indices=np.array(tri), vertex_colors=np.tile(color, (len(v), 1)))


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


def blueprint(models):
    plots = [
        rrb.TimeSeriesView(origin="/plots/speed", name="속도 [m/s]"),
        rrb.TimeSeriesView(origin="/plots/slip", name="슬립각 [deg]"),
        rrb.TimeSeriesView(origin="/plots/yaw_rate", name="요레이트 r: 실제 vs 1초 예측 [rad/s]"),
        rrb.TimeSeriesView(origin="/plots/lat_vel", name="횡속도 Uy: 실제 vs 1초 예측 [m/s]"),
        rrb.TimeSeriesView(origin="/plots/error", name="1초 뒤 위치 오차 [m]"),
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(origin="/world", name="Spielberg 1:10 · 차 추적 카메라", background=[22, 36, 26],
                                  line_grid=False, eye_controls=rrb.EyeControls3D(tracking_entity="/world/car")),
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="/world", contents=["/world/track/**", "/world/car/**", "/world/trail"],
                                      name="전체 서킷", background=[22, 36, 26], line_grid=False),
                    rrb.TextDocumentView(origin="/info", name="설명"),
                ),
                row_shares=[3, 1.2],
            ),
            rrb.Vertical(*plots),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--mu-dry", type=float, default=0.7)
    ap.add_argument("--mu-wet", type=float, default=0.55)
    ap.add_argument("--k-lat", type=float, default=0.75)
    ap.add_argument("--wet-len", type=float, default=25.0)
    ap.add_argument("--before", type=float, default=8.0)
    ap.add_argument("--after", type=float, default=10.0)
    ap.add_argument("--path-hz", type=int, default=20, help="예측 궤적을 초당 몇 번 그릴지")
    ap.add_argument("--out", default=os.path.join(ROOT, "viz", "out", "demo.rrd"))
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    tr, wet, sim = scenario(args)
    n = len(sim["t"])
    step = 100 // args.path_hz
    frames = np.arange(4, n - 101, step)
    paths, kinds = {}, []
    for p in args.ckpt:
        model, ck = load_checkpoint(p)
        paths[ck["kind"]] = predict_paths(model, sim, frames); kinds.append(ck["kind"])
    truth = truth_paths(sim, frames)
    fde = {k: np.linalg.norm(paths[k][:, -1] - truth[:, -1], axis=1) for k in paths}

    rr.init("vehicle_dynamics_demo")
    rr.save(args.out, default_blueprint=blueprint(kinds))

    # ---- 고정 요소
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    L, R = tr["left"], tr["right"]
    rr.log("/world/track/asphalt", strip_mesh(L, R, Z_ROAD, [52, 56, 64]), static=True)
    # 연석: 경계선 바깥쪽 얇은 띠 (빨강/흰색 줄무늬 대신 단색)
    def offset(line, other, d):
        v = line - other; v /= np.linalg.norm(v, axis=1, keepdims=True); return line + d * v
    rr.log("/world/track/kerb", rr.LineStrips3D([np.c_[np.r_[L, L[:1]], np.full(len(L) + 1, Z_LINE)],
                                                  np.c_[np.r_[R, R[:1]], np.full(len(R) + 1, Z_LINE)]],
                                                 colors=[[230, 230, 230]], radii=0.04), static=True)
    rr.log("/world/track/center", rr.LineStrips3D([np.c_[tr["center"], np.full(len(tr["center"]), Z_LINE)]],
                                                   colors=[[200, 200, 200, 90]], radii=0.01), static=True)
    w = (tr["s"] >= wet[0]) & (tr["s"] <= wet[1])
    idx = np.where(w)[0]
    if len(idx) > 1:
        # 젖은 구간: 주행 경로 인덱스를 경계선 인덱스로 맞춰서 채움(중심선 기반이라 인덱스가 같다)
        rr.log("/world/track/wet", rr.Mesh3D(
            vertex_positions=np.vstack([np.c_[L[idx], np.full(len(idx), Z_WET)], np.c_[R[idx], np.full(len(idx), Z_WET)]]),
            triangle_indices=np.array([[i, i + 1, len(idx) + i] for i in range(len(idx) - 1)] +
                                      [[i + 1, len(idx) + i + 1, len(idx) + i] for i in range(len(idx) - 1)]),
            vertex_colors=np.tile([47, 129, 247, 140], (2 * len(idx), 1))), static=True)
    # 차체: 1/10 RC카 크기 상자(길이 0.5 m) + 진행 방향 표시
    rr.log("/world/car/body", rr.Boxes3D(centers=[[0, 0, 0.07]], half_sizes=[[0.25, 0.14, 0.06]], colors=[CAR],
                                         fill_mode="solid"), static=True)
    rr.log("/world/car/nose", rr.Arrows3D(origins=[[0.0, 0, 0.14]], vectors=[[0.35, 0, 0]], colors=[[20, 20, 20]], radii=0.025),
           static=True)
    for k in paths:
        rr.log(f"/plots/error/{k}", rr.SeriesLines(colors=[RGB[k]], names=[LABELS[k]], widths=[2]), static=True)
        rr.log(f"/plots/yaw_rate/{k}", rr.SeriesLines(colors=[RGB[k]], names=[f"{LABELS[k]} 예측"], widths=[1.5]), static=True)
        rr.log(f"/plots/lat_vel/{k}", rr.SeriesLines(colors=[RGB[k]], names=[f"{LABELS[k]} 예측"], widths=[1.5]), static=True)
    for p_, nm in (("yaw_rate", "실제"), ("lat_vel", "실제")):
        rr.log(f"/plots/{p_}/truth", rr.SeriesLines(colors=[[230, 230, 230]], names=[nm], widths=[2.5]), static=True)
    rr.log("/plots/speed/ux", rr.SeriesLines(colors=[CAR], names=["속도"], widths=[2]), static=True)
    rr.log("/plots/speed/mu", rr.SeriesLines(colors=[[47, 129, 247]], names=["노면 μ ×10"], widths=[2]), static=True)
    rr.log("/plots/slip/beta", rr.SeriesLines(colors=[CAR], names=["차체 슬립 β"], widths=[2]), static=True)
    rr.log("/plots/slip/af", rr.SeriesLines(colors=[[255, 120, 120]], names=["앞바퀴 α_f"], widths=[1.5]), static=True)
    rr.log("/plots/slip/ar", rr.SeriesLines(colors=[[160, 220, 255]], names=["뒷바퀴 α_r"], widths=[1.5]), static=True)
    rr.log("/info", rr.TextDocument(
        "## 젖은 노면에서 차가 밀려날 때, 어느 모델이 1초 전에 알아채나\n\n"
        f"- 서킷: Spielberg (1:10), 소형차(F1TENTH) 시뮬레이션\n"
        f"- 파란 구역: 젖은 노면 μ {args.mu_dry} → {args.mu_wet}. 운전자는 모르고 같은 속도로 진입\n"
        "- **흰 선**: 실제로 갈 1초 궤적\n"
        + "".join(f"- **{LABELS[k]}** 선: 지금 시점에 예측한 1초 궤적\n" for k in paths)
        + "\n오른쪽 그래프의 예측값은 '지금 예측한 1초 뒤 값'을 지금 시각에 찍은 것이다.",
        media_type=rr.MediaType.MARKDOWN), static=True)

    # ---- 시간에 따라 움직이는 요소 (100 Hz)
    t0 = time.time()
    fpos = {f: i for i, f in enumerate(frames)}
    for s in range(n):
        rr.set_time("time", duration=float(sim["t"][s]))
        X, Y, psi = float(sim["X"][s]), float(sim["Y"][s]), float(sim["psi"][s])
        rr.log("/world/car", rr.Transform3D(translation=[X, Y, 0.0], rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=psi)))
        rr.log("/plots/speed/ux", rr.Scalars(sim["ux"][s]))
        rr.log("/plots/speed/mu", rr.Scalars(10 * sim["mu"][s]))
        rr.log("/plots/slip/beta", rr.Scalars(np.degrees(np.arctan2(sim["uy"][s], sim["ux"][s]))))
        rr.log("/plots/slip/af", rr.Scalars(np.degrees(sim["alpha_f"][s])))
        rr.log("/plots/slip/ar", rr.Scalars(np.degrees(sim["alpha_r"][s])))
        if s + 100 < n:  # 1초 뒤 실제 값 (예측과 같은 시각에 찍음)
            rr.log("/plots/yaw_rate/truth", rr.Scalars(sim["r"][s + 100]))
            rr.log("/plots/lat_vel/truth", rr.Scalars(sim["uy"][s + 100]))
        if s % 10 == 0:
            a0 = max(0, s - 300)
            rr.log("/world/trail", rr.LineStrips3D([np.c_[sim["X"][a0:s + 1], sim["Y"][a0:s + 1], np.full(s + 1 - a0, Z_PATH)]],
                                                   colors=[[255, 211, 61, 150]], radii=0.02))
        if s in fpos:
            fi = fpos[s]
            rr.log("/world/pred/truth", rr.LineStrips3D([np.c_[truth[fi], np.full(len(truth[fi]), Z_PATH + 0.01)]],
                                                        colors=[RGB["truth"]], radii=0.035))
            for k in paths:
                rr.log(f"/world/pred/{k}", rr.LineStrips3D([np.c_[paths[k][fi], np.full(len(paths[k][fi]), Z_PATH)]],
                                                           colors=[RGB[k]], radii=0.03))
                rr.log(f"/world/pred/{k}_end", rr.Points3D([[*paths[k][fi, -1], Z_PATH]], colors=[RGB[k]], radii=0.07))
                rr.log(f"/plots/error/{k}", rr.Scalars(fde[k][fi]))
    # 모델별 1초 뒤 r, Uy 예측값 (그 예측을 한 시각에 찍음 → 같은 시각의 '실제 1초 뒤 값'과 비교)
    for p in args.ckpt:
        model, ck = load_checkpoint(p); k = ck["kind"]
        rU = rollout_last(model, sim, frames)
        for fi, s in enumerate(frames):
            rr.set_time("time", duration=float(sim["t"][s]))
            rr.log(f"/plots/yaw_rate/{k}", rr.Scalars(rU[fi, 0]))
            rr.log(f"/plots/lat_vel/{k}", rr.Scalars(rU[fi, 1]))
    rr.disconnect()
    print(f"기록 {time.time() - t0:.0f}s -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
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
