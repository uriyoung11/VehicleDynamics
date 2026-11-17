"""MPC 주행 비교 영상. viz/demo_track.mp4와 같은 화면 구성(어두운 배경, 추격 화면 + 서킷·그래프·상태판).

  python -m control.mpc_video --mode A          # -> viz/out/mpc/mpc_A_panels.mp4
  python -m control.mpc_video --mode B          # -> viz/out/mpc/mpc_B_panels.mp4
  python -m control.mpc_video --mode A --layout single   # 모든 차를 한 화면에 -> viz/out/mpc/mpc_A_single.mp4

같은 트랙·같은 MPC에서 예측 모델만 바꾼 제어기마다 칸을 나눠, 각 칸이 그 차를 따라간다(차끼리 가리지 않게).
모든 칸에 정답 μ MPC(상한선) 차를 흰 반투명 고스트로 겹쳐 기준과 비교한다. 트랙을 벗어나거나 스핀·멈춘 차는 그 자리에 빨간 테두리와 ×로 남는다.
control.run_scene --mode A/B 결과(viz/out/mpc/<제어기>_<A|B>.npz)를 읽기만 한다. 같은 이름의 영상이 있으면 덮어쓰지 않는다.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms
from matplotlib.collections import PolyCollection

from viz.demo_track import load_track
from viz.demo_rerun import Margin
from control.run_scene import OUT, failure, scenario

# 어두운 배경용 색 (viz/demo_track과 같은 계열)
COL = {"mppi_oracle": "#ffffff", "mppi_dry": "#ffd33d", "mppi_m0": "#ff8a3d", "mppi_b1": "#4ea3ff",
       "mppi_m1": "#3ddc84", "mppi_m2": "#c77dff", "mppi_m3": "#ff6fb5", "pp": "#8b949e"}
LAB = {"mppi_oracle": "물리, 정답 μ (상한선)", "mppi_dry": "물리, 마른 μ만 앎", "mppi_m0": "M0 신경망", "mppi_b1": "B1 물리 모델",
       "mppi_m1": "M1 물리 잔차", "mppi_m2": "M2 타이어 학습", "mppi_m3": "M3 PINN", "pp": "pure pursuit"}
TITLE = {"A": ("MPC 방식 A: 같은 목표 속도를 주고, 예측 모델만 바꿨을 때",
               "목표 속도 = 마른 노면 한계 × 0.75 (모든 차 같음)   |   파란 띠: 젖은 노면 (μ 0.7 → 0.55)   |   ×: 트랙 이탈·스핀·멈춤"),
         "B": ("MPC 방식 B: 목표 속도 없이 '트랙 안에서 최대한 멀리'만 주고, 예측 모델만 바꿨을 때",
               "속도는 각 모델의 예측으로 MPC가 정한다   |   파란 띠: 젖은 노면 (μ 0.7 → 0.55)   |   ×: 트랙 이탈·스핀·멈춤")}


# 위에서 본 레이스카 (차 좌표계, x 앞쪽, 단위 m, 1/10 RC카 크기)
_BODY = np.array([[-0.22, -0.085], [0.08, -0.085], [0.20, -0.045], [0.27, 0.0], [0.20, 0.045], [0.08, 0.085], [-0.22, 0.085]])
_PARTS = [("wing_f", np.array([[0.22, -0.14], [0.27, -0.14], [0.27, 0.14], [0.22, 0.14]]), "body"),
          ("wing_r", np.array([[-0.28, -0.12], [-0.23, -0.12], [-0.23, 0.12], [-0.28, 0.12]]), "body"),
          *[(f"wheel{i}", np.array([[x - 0.05, y - 0.03], [x + 0.05, y - 0.03], [x + 0.05, y + 0.03], [x - 0.05, y + 0.03]]), "wheel")
            for i, (x, y) in enumerate([(0.15, 0.125), (0.15, -0.125), (-0.15, 0.13), (-0.15, -0.13)])],
          ("body", _BODY, "body"),
          ("cockpit", np.array([[-0.06, -0.035], [0.05, -0.03], [0.07, 0.0], [0.05, 0.03], [-0.06, 0.035]]), "cockpit")]


def car_patches(ax, color, alpha, z):
    """위에서 본 차 모양 패치들(차체, 앞뒤 날개, 바퀴 4개, 콕핏)."""
    ps = []
    for name, poly, kind in _PARTS:
        fc = {"body": color, "wheel": "#111111", "cockpit": "#1b1f24"}[kind]
        p = patches.Polygon(poly, closed=True, fc=fc, ec="#000000", lw=0.8 if kind != "wheel" else 0.4, alpha=alpha, zorder=z)
        ax.add_patch(p); ps.append(p)
    return ps


def place(ps, ax, X, Y, psi, dead):
    tf = transforms.Affine2D().rotate(psi).translate(X, Y) + ax.transData
    for p in ps:
        p.set_transform(tf)
        if dead:
            p.set_alpha(0.5)


def render_single(args, out, tr, wet, runs, fails, marg, fol, t_end):
    """한 화면 버전: 모든 차를 같은 추격 화면에 겹쳐 그린다.
    차끼리 가리지 않게: 반투명 차체 + 색 테두리, 차마다 이름표를 진행 방향 옆으로 펼쳐 놓고 가는 선으로 잇는다.
    실패한 차는 바닥 층으로 내리고 흐리게 남긴다."""
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    bg, asphalt, line = "#0e1116", "#2a2f38", "#c9d1d9"
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=bg)
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 0.02], height_ratios=[1.15, 1, 1], left=0.02, right=0.98, top=0.9, bottom=0.06, wspace=0.08, hspace=0.38)
    ax = fig.add_subplot(gs[:, 0]); axm = fig.add_subplot(gs[0, 1]); axv = fig.add_subplot(gs[1, 1]); axe = fig.add_subplot(gs[2, 1])
    for a in (ax, axm, axv, axe):
        a.set_facecolor(bg if a in (axv, axe) else "#16241a")
        for sp in a.spines.values(): sp.set_color("#30363d")
        a.tick_params(colors="#8b949e", labelsize=8)
    L, R = tr["left"], tr["right"]
    quads = [np.array([L[i], L[(i + 1) % len(L)], R[(i + 1) % len(R)], R[i]]) for i in range(len(L))]
    w = (tr["s"] >= wet[0]) & (tr["s"] <= wet[1])
    has_wet = args.mu_wet < 0.7 - 1e-6
    for a in (ax, axm):
        a.add_collection(PolyCollection(quads, facecolors=asphalt, edgecolors=asphalt, lw=0.5, zorder=0))
        a.plot(*L.T, color=line, lw=1.2 if a is ax else 0.6, zorder=1); a.plot(*R.T, color=line, lw=1.2 if a is ax else 0.6, zorder=1)
        a.plot(*tr["xy"].T, color="#ffffff", lw=0.6, ls=(0, (4, 6)), alpha=0.35, zorder=1)
        if has_wet:
            a.plot(*tr["xy"][w].T, color="#2f81f7", lw=26 if a is ax else 5, alpha=0.28, solid_capstyle="butt", zorder=1)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    axm.set_title(f"{args.track} 서킷 (1:10)", color="#e6edf3", fontsize=10)
    fig.text(0.02, 0.965, TITLE[args.mode][0], color="#e6edf3", fontsize=16, weight="bold")
    fig.text(0.02, 0.937, TITLE[args.mode][1], color="#8b949e", fontsize=10)

    w_t = fol["t"][fol["mu"] < 0.6]
    for a, key, name in ((axv, "ux", "속도 [m/s]"), (axe, None, "트랙 가장자리까지 여유 [m] (0 아래 = 이탈)")):
        for c, r in runs.items():
            fi = fails[c][0]; n = len(r["t"]) if fi is None else fi + 1
            y = r["ux"] if key else marg[c]
            a.plot(r["t"][:n], y[:n], color=COL[c], lw=1.4)
            if fi is not None:
                a.plot(r["t"][fi], y[fi], "x", color="#ff4d4d", ms=8, mew=2)
        if len(w_t):
            a.axvspan(w_t.min(), w_t.max(), color="#2f81f7", alpha=0.18)
        a.set_title(name, color="#e6edf3", fontsize=10); a.set_xlim(0, t_end)
    axe.axhline(0, color="#ff4d4d", lw=0.8, ls="--"); axe.set_ylim(-0.6, 1.2); axv.set_ylim(0, 7.8)
    marks = [a.axvline(0, color="#e6edf3", lw=0.8) for a in (axv, axe)]

    order = list(runs)
    cars, trails, minis, xs, tags, leads = {}, {}, {}, {}, {}, {}
    for k, c in enumerate(order):
        cars[c] = car_patches(ax, COL[c], 0.78, 8 + k * 0.01)
        trails[c], = ax.plot([], [], color=COL[c], lw=1.8, alpha=0.8, zorder=4)
        minis[c], = axm.plot([], [], "o", color=COL[c], ms=5, zorder=5)
        xs[c], = ax.plot([], [], "x", color="#ff4d4d", ms=14, mew=3, zorder=7)
        leads[c], = ax.plot([], [], color=COL[c], lw=0.9, alpha=0.9, zorder=9)
        tags[c] = ax.text(0, 0, "", color="#0e1116", fontsize=9, weight="bold", ha="center", va="center", zorder=10,
                          bbox=dict(fc=COL[c], ec="none", alpha=0.92, boxstyle="round,pad=0.25"))
    for c in order:
        tags[c].set_visible(args.labels); leads[c].set_visible(args.labels)
    hud = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left", color="#e6edf3", fontsize=11, linespacing=1.45,
                  bbox=dict(fc="#161b22", ec="#30363d", alpha=0.88, boxstyle="round,pad=0.5"), zorder=11)
    banner = ax.text(0.5, 0.05, "", transform=ax.transAxes, ha="center", color="#ffffff", fontsize=15, weight="bold",
                     bbox=dict(fc="#1f6feb", ec="none", alpha=0.0, boxstyle="round,pad=0.4"), zorder=11)
    leg = [plt.Line2D([], [], color=COL[c], lw=6, label=LAB[c]) for c in order]
    ax.legend(handles=leg, loc="upper right", fontsize=10, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3").set_zorder(12)
    short ={c: LAB[c].split(" ")[0] if c.startswith("mppi_m") or c == "mppi_b1" else {"mppi_oracle": "정답 μ", "mppi_dry": "마른 μ", "pp": "PP"}[c] for c in order}
    HW, HH = 5.0, 3.8

    def idx_at(r, t):
        return min(int(np.searchsorted(r["t"], t)), len(r["t"]) - 1)

    def update(k):
        t = k * args.speed / args.fps
        s = idx_at(fol, t)
        X0, Y0, psi0 = fol["X"][s], fol["Y"][s], fol["psi"][s]
        ax.set_xlim(X0 - HW, X0 + HW); ax.set_ylim(Y0 - HH, Y0 + HH)
        nrm = np.array([-np.sin(psi0), np.cos(psi0)])          # 진행 방향 왼쪽
        fwd = np.array([np.cos(psi0), np.sin(psi0)])
        rows = [f"t  {t:5.1f} s    노면 μ {fol['mu'][s]:.2f}", ""]
        # 이름표 자리: 차 바로 옆(달리는 차는 진행 방향 왼쪽, 실패한 차는 오른쪽)에 두고, 이름표끼리 가까우면 서로 밀어낸다
        pos, anchor = {}, {}
        for c in order:
            r = runs[c]; i = idx_at(r, t); fi = fails[c][0]
            dead = fi is not None and i >= fi
            i = fi if dead else i
            anchor[c] = np.array([r["X"][i], r["Y"][i]])
            pos[c] = anchor[c] + (-1.0 if dead else 1.0) * nrm * 0.7
        for _ in range(60):
            moved = False
            for a_ in range(len(order)):
                for b_ in range(a_ + 1, len(order)):
                    ca, cb = order[a_], order[b_]
                    d = pos[cb] - pos[ca]; dist = np.hypot(*d)
                    if dist < 0.5:
                        u = d / dist if dist > 1e-6 else fwd
                        push = (0.5 - dist) / 2 + 0.01
                        pos[ca] -= u * push; pos[cb] += u * push; moved = True
            if not moved:
                break
        for c in order:   # 화면 밖으로 나가지 않게
            pos[c] = np.clip(pos[c], [X0 - HW + 0.4, Y0 - HH + 0.9], [X0 + HW - 0.4, Y0 + HH - 0.3])
        for c in order:
            r = runs[c]; i = idx_at(r, t); fi, why = fails[c]
            dead = fi is not None and i >= fi
            if dead:
                i = fi
            X, Y = r["X"][i], r["Y"][i]
            a0 = max(0, i - 300)
            trails[c].set_data(r["X"][a0:i + 1], r["Y"][a0:i + 1])
            place(cars[c], ax, X, Y, r["psi"][i], dead)
            for p in cars[c]:
                p.set_zorder(5 if dead else 8 + order.index(c) * 0.01)
            xs[c].set_data(([X], [Y]) if dead else ([], []))
            minis[c].set_data([X], [Y])
            tags[c].set_position(pos[c]); tags[c].set_text(short[c] + (f" ×{why}" if dead else ""))
            leads[c].set_data([X, pos[c][0]], [Y, pos[c][1]])
            rows.append(f"{LAB[c]:<14s} " + (f"× {why} {r['t'][fi]:.1f}s" if dead else f"{r['ux'][i]:4.1f} m/s  여유 {marg[c][i]:+.2f} m"))
        hud.set_text("\n".join(rows))
        wet_now = fol["mu"][s] < 0.6
        banner.set_text(f"젖은 노면 구간  μ = {fol['mu'][s]:.2f}" if wet_now else ""); banner.get_bbox_patch().set_alpha(0.9 if wet_now else 0.0)
        for m_ in marks:
            m_.set_xdata([t])
        return []

    return fig, update, bg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["panels", "single"], default="panels",
                    help="panels: 모델마다 칸을 나눔, single: 모든 차를 한 화면에 (색으로 구분)")
    ap.add_argument("--labels", action="store_true", help="single: 차마다 이름표를 붙인다 (기본은 끔)")
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--ctrl", nargs="+", default=["mppi_oracle", "mppi_dry", "mppi_b1", "mppi_m0", "mppi_m1", "mppi_m2", "mppi_m3"])
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--tag", default="")
    ap.add_argument("--mu-wet", type=float, default=0.55)
    ap.add_argument("--wet-len", type=float, default=25.0)
    ap.add_argument("--follow", default="mppi_oracle", help="화면이 따라갈 제어기")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None, help="이 시각[s]의 한 장면을 PNG로도 저장 (docs/figures/mpc_<mode>[_tag]_frame.png)")
    args = ap.parse_args(argv)
    sfx = f"_{args.mode}{'_' + args.tag if args.tag else ''}"
    out = args.out or os.path.join(OUT, f"mpc_{args.mode}_{args.layout}{'_' + args.tag if args.tag else ''}.mp4")
    if os.path.exists(out):
        raise SystemExit(f"{out} 이 이미 있다. 덮어쓰지 않는다. --out 으로 새 이름을 주거나 직접 지운다.")

    tr = load_track(args.track)
    ns = argparse.Namespace(track=args.track, mu_dry=0.7, mu_wet=args.mu_wet, k_lat=0.75, wet_len=args.wet_len, before=8.0, after=10.0)
    _, _, wet, _, _ = scenario(ns)
    runs = {c: dict(np.load(os.path.join(OUT, f"{c}{sfx}.npz"))) for c in args.ctrl if os.path.exists(os.path.join(OUT, f"{c}{sfx}.npz"))}
    t_end = max(r["t"][-1] for r in runs.values())
    fails = {c: failure(r, tr, t_end) for c, r in runs.items()}
    margin = Margin(tr)
    marg = {c: margin(np.c_[r["X"], r["Y"]])[0] for c, r in runs.items()}
    fol = runs[args.follow]
    if args.layout == "single":
        fig, update, bg = render_single(args, out, tr, wet, runs, fails, marg, fol, t_end)
        return save(fig, update, bg, args, out, t_end)

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    bg, asphalt, line = "#0e1116", "#2a2f38", "#c9d1d9"
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=bg)
    outer = fig.add_gridspec(1, 2, width_ratios=[2.45, 1], left=0.015, right=0.985, top=0.855, bottom=0.06, wspace=0.07)
    gl = outer[0].subgridspec(2, 4, wspace=0.05, hspace=0.22)
    gr = outer[1].subgridspec(3, 1, height_ratios=[1.1, 1, 1], hspace=0.42)
    order = [c for c in ["mppi_oracle", "mppi_dry", "mppi_b1", "mppi_m0", "mppi_m1", "mppi_m2", "mppi_m3", "pp"] if c in runs][:7]
    panels = {c: fig.add_subplot(gl[i // 4, i % 4]) for i, c in enumerate(order)}
    axi = fig.add_subplot(gl[1, 3]) if len(order) == 7 else None
    axm, axv, axe = fig.add_subplot(gr[0]), fig.add_subplot(gr[1]), fig.add_subplot(gr[2])
    for a in [*panels.values(), axm, axv, axe] + ([axi] if axi else []):
        a.set_facecolor(bg if a in (axv, axe, axi) else "#16241a")
        for sp in a.spines.values(): sp.set_color("#30363d")
        a.tick_params(colors="#8b949e", labelsize=8)
    L, R = tr["left"], tr["right"]
    quads = [np.array([L[i], L[(i + 1) % len(L)], R[(i + 1) % len(R)], R[i]]) for i in range(len(L))]
    w = (tr["s"] >= wet[0]) & (tr["s"] <= wet[1])
    has_wet = args.mu_wet < 0.7 - 1e-6
    for a in [*panels.values(), axm]:
        big = a is not axm
        a.add_collection(PolyCollection(quads, facecolors=asphalt, edgecolors=asphalt, lw=0.5, zorder=0))
        a.plot(*L.T, color=line, lw=1.0 if big else 0.6, zorder=1); a.plot(*R.T, color=line, lw=1.0 if big else 0.6, zorder=1)
        a.plot(*tr["xy"].T, color="#ffffff", lw=0.5, ls=(0, (4, 6)), alpha=0.3, zorder=1)
        if has_wet:
            a.plot(*tr["xy"][w].T, color="#2f81f7", lw=14 if big else 5, alpha=0.28, solid_capstyle="butt", zorder=1)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    axm.set_title(f"{args.track} 서킷 (1:10)", color="#e6edf3", fontsize=10)
    fig.text(0.015, 0.962, TITLE[args.mode][0], color="#e6edf3", fontsize=16, weight="bold")
    fig.text(0.015, 0.932, TITLE[args.mode][1] + "   |   흰 반투명 차: 정답 μ MPC (기준)", color="#8b949e", fontsize=10)

    # 그래프: 속도, 가장자리 여유 (모든 제어기)
    w_t = fol["t"][fol["mu"] < 0.6]
    for a, key, name in ((axv, "ux", "속도 [m/s]"), (axe, None, "트랙 가장자리까지 여유 [m] (0 아래 = 이탈)")):
        for c, r in runs.items():
            fi = fails[c][0]; n = len(r["t"]) if fi is None else fi + 1
            y = r["ux"] if key else marg[c]
            a.plot(r["t"][:n], y[:n], color=COL[c], lw=1.4)
            if fi is not None:
                a.plot(r["t"][fi], y[fi], "x", color="#ff4d4d", ms=8, mew=2)
        if len(w_t):
            a.axvspan(w_t.min(), w_t.max(), color="#2f81f7", alpha=0.18)
        a.set_title(name, color="#e6edf3", fontsize=10); a.set_xlim(0, t_end)
    axe.axhline(0, color="#ff4d4d", lw=0.8, ls="--"); axe.set_ylim(-0.6, 1.2); axv.set_ylim(0, 7.8)
    marks = [a.axvline(0, color="#e6edf3", lw=0.8) for a in (axv, axe)]

    # 패널마다: 자기 차(+꼬리), 기준 고스트(정답 μ MPC), 실패 표시
    ref = runs.get("mppi_oracle")
    own, ghost, trails, gtrails, xs, titles, minis = {}, {}, {}, {}, {}, {}, {}
    for c, a in panels.items():
        own[c] = car_patches(a, COL[c], 1.0, 8)
        trails[c], = a.plot([], [], color=COL[c], lw=2.0, alpha=0.85, zorder=4)
        xs[c], = a.plot([], [], "x", color="#ff4d4d", ms=18, mew=4, zorder=10)
        if ref is not None and c != "mppi_oracle":
            ghost[c] = car_patches(a, "#ffffff", 0.35, 7)
            gtrails[c], = a.plot([], [], color="#ffffff", lw=1.2, alpha=0.35, ls="--", zorder=3)
        titles[c] = a.set_title("", color=COL[c], fontsize=11, weight="bold", loc="left")
        minis[c], = axm.plot([], [], "o", color=COL[c], ms=5, zorder=5)
    info = None
    if axi is not None:
        axi.axis("off")
        info = axi.text(0.04, 0.96, "", transform=axi.transAxes, va="top", ha="left", color="#e6edf3", fontsize=12, linespacing=1.6)
        legend = [plt.Line2D([], [], color=COL[c], lw=6, label=LAB[c]) for c in order]
        axi.legend(handles=legend, loc="lower left", fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    banner = fig.text(0.34, 0.012, "", ha="center", color="#ffffff", fontsize=14, weight="bold",
                      bbox=dict(fc="#1f6feb", ec="none", alpha=0.0, boxstyle="round,pad=0.35"))
    HW, HH = 1.9, 2.55   # 패널 반폭·반높이 [m]

    def idx_at(r, t):
        return min(int(np.searchsorted(r["t"], t)), len(r["t"]) - 1)

    def state(c, t):
        r = runs[c]; i = idx_at(r, t); fi, why = fails[c]
        dead = fi is not None and i >= fi
        return r, (fi if dead else i), dead, why

    def update(k):
        t = k * args.speed / args.fps
        s = idx_at(fol, t)
        n_ok = 0
        for c, a in panels.items():
            r, i, dead, why = state(c, t)
            X, Y = r["X"][i], r["Y"][i]
            a.set_xlim(X - HW, X + HW); a.set_ylim(Y - HH, Y + HH)
            a0 = max(0, i - 300)
            trails[c].set_data(r["X"][a0:i + 1], r["Y"][a0:i + 1])
            place(own[c], a, X, Y, r["psi"][i], dead)
            xs[c].set_data(([X], [Y]) if dead else ([], []))
            if c in ghost:
                rr_, j, _, _ = state("mppi_oracle", t)
                place(ghost[c], a, rr_["X"][j], rr_["Y"][j], rr_["psi"][j], False)
                b0 = max(0, j - 300); gtrails[c].set_data(rr_["X"][b0:j + 1], rr_["Y"][b0:j + 1])
            minis[c].set_data([X], [Y])
            titles[c].set_text(f"{LAB[c]}\n× {why} ({r['t'][i]:.1f}s)" if dead else f"{LAB[c]}\n{r['ux'][i]:.1f} m/s   여유 {marg[c][i]:+.2f} m")
            n_ok += not dead
        if info is not None:
            info.set_text(f"t = {t:5.1f} s\n노면 μ = {fol['mu'][s]:.2f}\n달리는 차 {n_ok} / {len(panels)}")
        wet_now = fol["mu"][s] < 0.6
        banner.set_text(f"젖은 노면 구간  μ = {fol['mu'][s]:.2f}" if wet_now else ""); banner.get_bbox_patch().set_alpha(0.9 if wet_now else 0.0)
        for m_ in marks:
            m_.set_xdata([t])
        return []

    return save(fig, update, bg, args, out, t_end)


def save(fig, update, bg, args, out, t_end):
    n_frames = int(t_end / args.speed * args.fps) + 1
    if args.png is not None:
        update(int(float(args.png) / args.speed * args.fps))
        lay = "" if args.layout == "panels" else f"_{args.layout}"
        p = os.path.join(os.path.dirname(OUT), "..", "..", "docs", "figures", f"mpc_{args.mode}{lay}{'_' + args.tag if args.tag else ''}_frame.png")
        fig.savefig(os.path.normpath(p), facecolor=bg); print("saved", os.path.normpath(p))
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps)
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    ani.save(out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=6000), savefig_kwargs=dict(facecolor=bg))
    print(f"{n_frames} 프레임 -> {out}")


if __name__ == "__main__":
    main()
