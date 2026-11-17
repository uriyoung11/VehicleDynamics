"""MPC 방식 A·B 주행 결과를 Rerun 기록(.rrd)으로 만든다. 팀 분석용 (발표용은 Blender).

  python -m control.rerun_mpc                       # viz/out/mpc/*_A.npz, *_B.npz -> viz/out/mpc/mpc_AB.rrd
  rerun viz/out/mpc/mpc_AB.rrd                      # 화면이 깨지면: rerun --renderer gl viz/out/mpc/mpc_AB.rrd

화면
  - 왼쪽 3D: 방식 A(목표 속도 추종), 오른쪽 3D: 방식 B(진행 최대화). 같은 트랙에 모든 제어기 차를 겹쳐 그린다.
      차 색 = 제어기, 꼬리 = 최근 1.5초 궤적, 빨간 ✕ = 실패 지점(트랙 이탈 / 스핀 / 멈춤)
  - 아래 그래프: 방식별 속도, 가장자리까지 여유(0 아래 = 이탈). 범례를 눌러 제어기를 켜고 끌 수 있다.
  - 상태판: 제어기별 결과(완주 / 실패 이유와 시각)
시간 슬라이더로 앞뒤로 움직이며 볼 수 있다. control.run_scene --mode A/B 결과를 읽기만 한다.
"""
import argparse, os
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from viz.demo_track import load_track
from viz.demo_rerun import strip_mesh, Margin
from control.run_scene import COLORS, LABEL, OUT, failure, scenario

GRASS = [24, 40, 28]
MODES = {"A": "방식 A: 목표 속도 추종 (속도는 사람이 정함)", "B": "방식 B: 진행 최대화 (속도는 모델 예측으로 정해짐)"}


def rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def log_track(root, tr, wet):
    rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    L, R = tr["left"], tr["right"]
    rr.log(f"{root}/track/asphalt", strip_mesh(L, R, 0.0, [58, 60, 66]), static=True)
    for side, line in (("l", L), ("r", R)):
        rr.log(f"{root}/track/edge_{side}", rr.LineStrips3D([np.c_[np.r_[line, line[:1]], np.full(len(line) + 1, 0.01)]],
                                                            colors=[[235, 235, 235]], radii=0.03), static=True)
    idx = np.where((tr["s"] >= wet[0]) & (tr["s"] <= wet[1]))[0]
    m = len(idx)
    rr.log(f"{root}/track/wet", rr.Mesh3D(
        vertex_positions=np.vstack([np.c_[L[idx], np.full(m, 0.004)], np.c_[R[idx], np.full(m, 0.004)]]),
        triangle_indices=np.array([[i, i + 1, m + i] for i in range(m - 1)] + [[i + 1, m + i + 1, m + i] for i in range(m - 1)]),
        vertex_colors=np.tile([47, 129, 247, 150], (2 * m, 1))), static=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", nargs="+", default=["pp", "mppi_dry", "mppi_oracle", "mppi_b1", "mppi_m0", "mppi_m1", "mppi_m2", "mppi_m3"])
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--tag", default="")
    ap.add_argument("--mu-wet", type=float, default=0.55)
    ap.add_argument("--wet-len", type=float, default=25.0)
    ap.add_argument("--every", type=int, default=2, help="기록 간격 (0.01초 단위)")
    ap.add_argument("--out", default=os.path.join(OUT, "mpc_AB.rrd"))
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} 이 이미 있다. 덮어쓰지 않는다. --out 으로 새 이름을 주거나 직접 지운다.")

    tr = load_track(args.track)
    ns = argparse.Namespace(track=args.track, mu_dry=0.7, mu_wet=args.mu_wet, k_lat=0.75, wet_len=args.wet_len, before=8.0, after=10.0)
    _, _, wet, _, _ = scenario(ns)
    margin = Margin(tr)
    runs = {}
    for mode in MODES:
        for c in args.ctrl:
            f = os.path.join(OUT, f"{c}_{mode}{'_' + args.tag if args.tag else ''}.npz")
            if os.path.exists(f):
                runs[(mode, c)] = dict(np.load(f))
    t_end = max(r["t"][-1] for r in runs.values())
    fails = {k: failure(r, tr, t_end) for k, r in runs.items()}

    views = [rrb.Spatial3DView(origin=f"/{mode}", name=MODES[mode], background=GRASS, line_grid=False) for mode in MODES]
    plots = rrb.Tabs(*[rrb.TimeSeriesView(origin=f"/plots/{q}/{mode}", name=f"{mode}: {name}")
                       for q, name in (("speed", "속도 [m/s]"), ("margin", "가장자리까지 여유 [m] (0 아래 = 이탈)")) for mode in MODES])
    bp = rrb.Blueprint(rrb.Vertical(rrb.Horizontal(*views), rrb.Horizontal(plots, rrb.TextDocumentView(origin="/hud", name="결과"),
                                                                           column_shares=[3, 1]), row_shares=[2, 1]),
                       collapse_panels=True)
    rr.init("mpc_ab")
    rr.save(args.out, default_blueprint=bp)

    lines = []
    for mode in MODES:
        log_track(f"/{mode}", tr, wet)
        lines.append(f"### {MODES[mode]}")
        for c in args.ctrl:
            if (mode, c) not in runs:
                continue
            col, (fi, why) = rgb(COLORS[c]), fails[(mode, c)]
            rr.log(f"/{mode}/car/{c}/body", rr.Boxes3D(centers=[[0.05, 0, 0.06]], half_sizes=[[0.25, 0.12, 0.05]], colors=[col],
                                                     labels=[LABEL[c]], show_labels=False), static=True)
            for q in ("speed", "margin"):
                rr.log(f"/plots/{q}/{mode}/{c}", rr.SeriesLines(colors=[col], names=[LABEL[c]], widths=[2]), static=True)
            res = "완주" if fi is None else f"{why} ({runs[(mode, c)]['t'][fi]:.2f}초)"
            lines.append(f"- **{LABEL[c]}**: {res}")
        for q in ("margin",):
            rr.log(f"/plots/{q}/{mode}/edge", rr.SeriesLines(colors=[[240, 70, 60]], names=["트랙 경계"], widths=[1]), static=True)
    rr.log("/hud", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)

    for (mode, c), r in runs.items():
        fi, why = fails[(mode, c)]
        m, _ = margin(np.c_[r["X"], r["Y"]])
        last = len(r["t"]) - 1 if fi is None else fi
        for s in range(0, last + 1, args.every):
            rr.set_time("time", duration=float(r["t"][s]))
            X, Y, psi = r["X"][s], r["Y"][s], r["psi"][s]
            rr.log(f"/{mode}/car/{c}", rr.Transform3D(translation=[X, Y, 0.0], rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=psi)))
            if s % 10 == 0:
                a = max(0, s - 150)
                rr.log(f"/{mode}/trail/{c}", rr.LineStrips3D([np.c_[r["X"][a:s + 1], r["Y"][a:s + 1], np.full(s + 1 - a, 0.02)]],
                                                              colors=[rgb(COLORS[c])], radii=0.03))
            rr.log(f"/plots/speed/{mode}/{c}", rr.Scalars(r["ux"][s]))
            rr.log(f"/plots/margin/{mode}/{c}", rr.Scalars(m[s]))
        if fi is not None:
            rr.set_time("time", duration=float(r["t"][fi]))
            rr.log(f"/{mode}/fail/{c}", rr.Points3D([[r["X"][fi], r["Y"][fi], 0.15]], colors=[[240, 40, 40]], radii=0.25,
                                                    labels=[f"{LABEL[c]} {why}"], show_labels=True))
    for mode in MODES:
        for s in range(0, int(t_end / 0.01) + 1, 10):
            rr.set_time("time", duration=s * 0.01)
            rr.log(f"/plots/margin/{mode}/edge", rr.Scalars(0.0))
    print("saved", args.out)


if __name__ == "__main__":
    main()
