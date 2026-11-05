"""발표용 2D 데모: 실제 서킷(Spielberg, 1:10) 위를 소형차가 달리고, 차 앞에 모델별 1초 예측 궤적을 그린다.

시나리오: 마른 노면(μ 0.7)을 기준으로 속도를 정한 운전자가, 가장 빠른 코너에 있는 젖은 구간(μ 0.4)에
모르고 들어간다. 이때 각 모델의 1초 예측이 실제 궤적과 얼마나 벌어지는지 본다.

  python -m viz.demo_track --ckpt checkpoints/f1tenth_v2_trial/m0_s0.pt checkpoints/f1tenth_v2_trial/b1_s0.pt

출력: viz/out/demo_track.mp4, docs/figures/demo_track_frame.png
트랙: data/tracks/Spielberg_centerline.csv (github.com/f1tenth/f1tenth_racetracks)
"""
import argparse, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms

from physics import F1TENTH, P2, rk4_step, derivatives
from models.baselines import load_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 9.81
COLORS = {"truth": "#ffffff", "m0": "#ff8a3d", "b1": "#4ea3ff", "m1": "#3ddc84", "m2": "#c77dff"}
LABELS = {"m0": "M0 신경망", "b1": "B1 물리 모델", "m1": "M1 물리 잔차", "m2": "M2 타이어 학습"}


# ---------------------------------------------------------------- 트랙
def smooth_closed(xy, sigma=3):
    """닫힌 곡선을 가우시안으로 매끄럽게(GPS 점 잡음 제거). sigma는 점 개수 단위(점 간격 약 0.4 m)."""
    if sigma <= 0:
        return xy
    k = np.arange(-3 * sigma, 3 * sigma + 1); w = np.exp(-0.5 * (k / sigma) ** 2); w /= w.sum()
    n = len(xy)
    return np.stack([sum(wi * np.roll(xy[:, d], -ki) for ki, wi in zip(k, w)) for d in range(2)], 1)


def load_track(name="Spielberg"):
    d = os.path.join(ROOT, "data", "tracks")
    f = os.path.join(d, f"{name}_centerline.csv")
    if not os.path.exists(f):  # data/는 git에 없으므로 처음 실행 시 f1tenth_racetracks에서 내려받는다
        import urllib.request
        os.makedirs(d, exist_ok=True)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/f1tenth/f1tenth_racetracks/main/{name}/{name}_centerline.csv", f)
    cl = np.loadtxt(os.path.join(d, f"{name}_centerline.csv"), delimiter=",", comments="#")
    # 주행 경로: 중심선(양쪽 여유 1.1 m). 최적 주행선(raceline)은 가장자리에 붙어 있어 데모용으로는 여유가 없다
    xy = smooth_closed(cl[:, :2], sigma=3)
    seg = np.linalg.norm(np.diff(xy, axis=0, append=xy[:1]), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)[:-1]])
    # 곡률은 위치에서 다시 계산(부호 규약 차이를 피하려고)
    dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    kappa = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    kappa = np.convolve(np.r_[kappa[-5:], kappa, kappa[:5]], np.ones(11) / 11, "same")[5:-5]
    # 트랙 경계
    c = cl[:, :2]; t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    left, right = c + n * cl[:, 3:4], c - n * cl[:, 2:3]
    return dict(xy=xy, s=s, kappa=kappa, length=s[-1] + seg[-1], center=c, left=left, right=right)


def speed_profile(tr, mu, k_lat=0.75, a_acc=0.35, a_brk=0.6, v_max=7.0):
    """곡률 한계 속도 + 앞뒤 가감속 한계(한 바퀴 순환)."""
    v = np.minimum(v_max, np.sqrt(k_lat * mu * G / np.maximum(np.abs(tr["kappa"]), 1e-4)))
    ds = np.diff(tr["s"], append=tr["length"])
    for _ in range(2):
        for i in range(len(v)):  # 가속 한계
            j = (i + 1) % len(v); v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_acc * mu * G * ds[i]))
        for i in range(len(v) - 1, -1, -1):  # 제동 한계
            j = (i + 1) % len(v); v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_brk * mu * G * ds[i]))
    return v


# ---------------------------------------------------------------- 시뮬레이션
def simulate(tr, v_ref, mu_of_s, duration, p=F1TENTH, dt_ctrl=0.01, dt_sim=0.001, start=0, ld0=1.0, ld_k=0.5):
    """주행선 start 지점에서 v_ref 속도로 출발."""
    xy, n = tr["xy"], len(tr["xy"])
    psi0 = np.arctan2(*(xy[(start + 1) % n] - xy[start])[::-1])
    x = torch.tensor([[xy[start, 0], xy[start, 1], psi0, v_ref[start], 0.0, 0.0]], dtype=torch.float64)
    delta, idx, rec = 0.0, start, []
    for k in range(int(duration / dt_ctrl)):
        X, Y, psi, ux = (float(v) for v in x[0, :4])
        # 가장 가까운 주행선 점(앞쪽 창에서만 찾음)
        win = (idx + np.arange(-5, 60)) % n
        d2 = np.sum((xy[win] - [X, Y]) ** 2, 1)
        idx = int(win[np.argmin(d2)])
        if d2.min() > 1.0:  # 경로를 놓치면 전체에서 다시 찾음
            idx = int(np.argmin(np.sum((xy - [X, Y]) ** 2, 1)))
        # pure pursuit
        Ld = ld0 + ld_k * ux  # 앞을 보는 거리. 짧으면 직선에서 좌우로 흔들림
        j = idx
        while np.hypot(*(xy[j] - [X, Y])) < Ld:
            j = (j + 1) % n
        dxl = (xy[j, 0] - X) * np.cos(psi) + (xy[j, 1] - Y) * np.sin(psi)
        dyl = -(xy[j, 0] - X) * np.sin(psi) + (xy[j, 1] - Y) * np.cos(psi)
        d_cmd = np.clip(np.arctan2(2 * p.L * dyl, dxl ** 2 + dyl ** 2), -p.delta_max, p.delta_max)
        delta += np.clip(d_cmd - delta, -p.delta_rate_max * dt_ctrl, p.delta_rate_max * dt_ctrl)
        mu = mu_of_s(tr["s"][idx])
        fx = p.m * 2.0 * (v_ref[idx] - ux) + (p.c0 + p.c1 * ux + p.c2 * ux ** 2)
        fx = float(np.clip(fx, -mu * p.m * G, min(p.power_max / max(ux, 0.5), mu * p.m * G)))
        u = torch.tensor([[delta, fx]], dtype=torch.float64)
        mu_t = torch.tensor([mu], dtype=torch.float64)
        xd, f = derivatives(x, u, p, mu_t, P2, return_forces=True)
        rec.append([k * dt_ctrl, *x[0].tolist(), delta, fx, float(f["fxf"][0]), float(f["ax"][0]), float(f["ay"][0]),
                    float(f["alpha_f"][0]), float(f["alpha_r"][0]), mu, idx])
        for _ in range(int(round(dt_ctrl / dt_sim))):
            x = rk4_step(x, u, p, mu_t, dt_sim, P2)
    cols = ["t", "X", "Y", "psi", "ux", "uy", "r", "delta", "fx", "fxf", "ax", "ay", "alpha_f", "alpha_r", "mu", "idx"]
    a = np.array(rec)
    return {c: a[:, i] for i, c in enumerate(cols)}


# ---------------------------------------------------------------- 모델 예측
@torch.no_grad()
def predict_paths(model, sim, starts, H=100, T=4, long_scale=F1TENTH.m * 100, dt=0.01):
    """평가 코드와 같은 롤아웃(r, Uy는 모델 예측, Ux·조향·구동력은 실제 값)을 하고,
    현재 위치·방향에서 적분해 전역 좌표 궤적으로 바꾼다."""
    F = torch.as_tensor(np.stack([sim["r"], sim["uy"], sim["ux"], sim["delta"], sim["fxf"] / long_scale], 1))
    hist = torch.stack([F[s - T + 1:s + 1] for s in starts])     # 현재 시점 s까지
    P = np.zeros((len(starts), H, 2))
    for k in range(H):
        pred = model(hist)
        P[:, k] = pred.numpy()
        nxt = torch.as_tensor(np.clip(np.array(starts) + k + 1, 0, len(F) - 1))
        hist = torch.cat([hist[:, 1:], torch.cat([pred, F[nxt, 2:]], -1)[:, None]], 1)
    paths = np.zeros((len(starts), H + 1, 2))
    for b, s in enumerate(starts):
        X, Y, psi = sim["X"][s], sim["Y"][s], sim["psi"][s]
        paths[b, 0] = X, Y
        for k in range(H):
            # 시각 s+k의 상태로 s+k -> s+k+1 구간을 적분. k=0은 실제 값, 그 뒤는 모델 예측
            ux = sim["ux"][min(s + k, len(F) - 1)]
            uy, r = (sim["uy"][s], sim["r"][s]) if k == 0 else (P[b, k - 1, 1], P[b, k - 1, 0])
            X += dt * (ux * np.cos(psi) - uy * np.sin(psi)); Y += dt * (ux * np.sin(psi) + uy * np.cos(psi)); psi += dt * r
            paths[b, k + 1] = X, Y
    return paths


def truth_paths(sim, starts, H=100):
    n = len(sim["X"])
    return np.stack([np.stack([sim["X"][np.clip(np.arange(s, s + H + 1), 0, n - 1)],
                               sim["Y"][np.clip(np.arange(s, s + H + 1), 0, n - 1)]], 1) for s in starts])


# ---------------------------------------------------------------- 그리기
def render(tr, sim, frames, paths, truth, wet, out_mp4, out_png, fps=20, png_frame=None):
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    bg, asphalt, line = "#0e1116", "#2a2f38", "#c9d1d9"
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=bg)
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 0.02], height_ratios=[1.15, 1, 1], left=0.02, right=0.98, top=0.93, bottom=0.06, wspace=0.08, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0]); axm = fig.add_subplot(gs[0, 1]); axf = fig.add_subplot(gs[1, 1]); axe = fig.add_subplot(gs[2, 1])
    for a in (ax, axm, axf, axe):
        a.set_facecolor(bg if a in (axf, axe) else "#16241a")  # 트랙 밖은 잔디(런오프) 색
        for sp in a.spines.values(): sp.set_color("#30363d")
        a.tick_params(colors="#8b949e", labelsize=8)

    from matplotlib.collections import PolyCollection
    L, R = tr["left"], tr["right"]
    quads = [np.array([L[i], L[(i + 1) % len(L)], R[(i + 1) % len(R)], R[i]]) for i in range(len(L))]
    for a in (ax, axm):
        a.add_collection(PolyCollection(quads, facecolors=asphalt, edgecolors=asphalt, lw=0.5, zorder=0))
        a.plot(*tr["left"].T, color=line, lw=1.2 if a is ax else 0.6, zorder=1); a.plot(*tr["right"].T, color=line, lw=1.2 if a is ax else 0.6, zorder=1)
        a.plot(*tr["xy"].T, color="#ffffff", lw=0.6, ls=(0, (4, 6)), alpha=0.35, zorder=1)
        w = (tr["s"] >= wet[0]) & (tr["s"] <= wet[1])
        a.plot(*tr["xy"][w].T, color="#2f81f7", lw=22 if a is ax else 5, alpha=0.28, solid_capstyle="butt", zorder=1)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    axm.set_title("Spielberg 서킷 (1:10)", color="#e6edf3", fontsize=10)
    fig.text(0.02, 0.965, "젖은 노면에서 차가 밀려날 때, 어느 모델이 1초 전에 알아채나", color="#e6edf3", fontsize=16, weight="bold")
    fig.text(0.02, 0.937, "흰색: 실제로 갈 궤적   |   선: 각 모델이 지금 시점에 예측한 1초 궤적   |   파란 띠: 젖은 노면 (μ 0.7 → 0.55)   |   초록: 트랙 밖", color="#8b949e", fontsize=10)

    # 마찰원
    th = np.linspace(0, 2 * np.pi, 200)
    axf.plot(np.cos(th), np.sin(th), color="#8b949e", lw=1); axf.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color="#30363d", lw=0.8)
    axf.set_xlim(-1.3, 1.3); axf.set_ylim(-1.3, 1.3); axf.set_aspect("equal")
    axf.set_title("마찰원: 가속도 / (μ·g)", color="#e6edf3", fontsize=10); axf.set_xlabel("횡", color="#8b949e", fontsize=8); axf.set_ylabel("종", color="#8b949e", fontsize=8)
    fc_trail, = axf.plot([], [], color="#ffd33d", lw=1, alpha=0.5); fc_dot, = axf.plot([], [], "o", color="#ffd33d", ms=8)

    # 1초 뒤 위치 오차
    tf = sim["t"][frames]
    fde = {k: np.linalg.norm(paths[k][:, -1] - truth[:, -1], axis=1) for k in paths}
    for k, v in fde.items():
        axe.plot(tf, v, color=COLORS[k], lw=1.6, label=LABELS[k])
    w_t = sim["t"][(tr["s"][sim["idx"].astype(int)] >= wet[0]) & (tr["s"][sim["idx"].astype(int)] <= wet[1])]
    if len(w_t):
        axe.axvspan(w_t.min(), w_t.max(), color="#2f81f7", alpha=0.18)
    axe.set_title("1초 뒤 위치 오차 [m]", color="#e6edf3", fontsize=10)
    axe.legend(fontsize=8, facecolor=bg, edgecolor="#30363d", labelcolor="#e6edf3", loc="upper left")
    axe.set_xlim(tf[0], tf[-1]); axe.set_ylim(0, max(1.0, 1.1 * max(np.percentile(v, 99) for v in fde.values())))
    tmark = axe.axvline(tf[0], color="#e6edf3", lw=0.8)

    # 움직이는 요소
    trail_full, = ax.plot([], [], color="#ffd33d", lw=1.5, alpha=0.6, zorder=3)
    tl_truth, = ax.plot([], [], color=COLORS["truth"], lw=3, zorder=5, solid_capstyle="round")
    pl = {k: ax.plot([], [], color=COLORS[k], lw=2.4, zorder=6)[0] for k in paths}
    end = {k: ax.plot([], [], "o", color=COLORS[k], ms=7, zorder=7)[0] for k in paths}
    car = patches.FancyBboxPatch((-0.25, -0.14), 0.5, 0.28, boxstyle="round,pad=0.02", fc="#ffd33d", ec="#000000", lw=1.2, zorder=8)
    ax.add_patch(car)
    mini, = axm.plot([], [], "o", color="#ffd33d", ms=6, zorder=5)
    hud = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left", color="#e6edf3", fontsize=12, linespacing=1.4,
                  bbox=dict(fc="#161b22", ec="#30363d", alpha=0.85, boxstyle="round,pad=0.5"), zorder=10)
    banner = ax.text(0.5, 0.06, "", transform=ax.transAxes, ha="center", color="#ffffff", fontsize=15, weight="bold",
                     bbox=dict(fc="#1f6feb", ec="none", alpha=0.0, boxstyle="round,pad=0.4"), zorder=10)
    leg = [plt.Line2D([], [], color=COLORS["truth"], lw=3, label="실제 궤적")] + [plt.Line2D([], [], color=COLORS[k], lw=2.4, label=LABELS[k]) for k in paths]
    ax.legend(handles=leg, loc="upper right", fontsize=10, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

    def update(fi):
        s = frames[fi]
        X, Y, psi = sim["X"][s], sim["Y"][s], sim["psi"][s]
        ax.set_xlim(X - 5.5, X + 5.5); ax.set_ylim(Y - 4.2, Y + 4.2)
        a0 = max(0, s - 300); trail_full.set_data(sim["X"][a0:s + 1], sim["Y"][a0:s + 1])
        tl_truth.set_data(*truth[fi].T)
        for k in paths:
            pl[k].set_data(*paths[k][fi].T); end[k].set_data([paths[k][fi, -1, 0]], [paths[k][fi, -1, 1]])
        car.set_transform(transforms.Affine2D().rotate(psi).translate(X, Y) + ax.transData)
        mini.set_data([X], [Y])
        mu = sim["mu"][s]
        fx_, fy_ = sim["ax"][s] / (mu * G), sim["ay"][s] / (mu * G)
        b0 = max(0, s - 100); fc_trail.set_data(sim["ay"][b0:s + 1] / (sim["mu"][b0:s + 1] * G), sim["ax"][b0:s + 1] / (sim["mu"][b0:s + 1] * G))
        fc_dot.set_data([fy_], [fx_])
        beta = np.degrees(np.arctan2(sim["uy"][s], sim["ux"][s]))
        hud.set_text(f"t      {sim['t'][s]:5.1f} s\n속도   {sim['ux'][s]:5.2f} m/s\n조향   {np.degrees(sim['delta'][s]):5.1f}°\n"
                     f"차체슬립 {beta:5.1f}°\n뒷바퀴슬립 {np.degrees(abs(sim['alpha_r'][s])):4.1f}°\n노면 μ  {mu:4.2f}\n"
                     + "\n".join(f"{LABELS[k][:2]} 1초오차 {fde[k][fi]:4.2f} m" for k in paths))
        wet_now = mu < 0.6
        banner.set_text(f"젖은 노면 구간  μ = {mu:.2f}" if wet_now else ""); banner.get_bbox_patch().set_alpha(0.9 if wet_now else 0.0)
        tmark.set_xdata([sim["t"][s]])
        return []

    if png_frame is not None:
        update(png_frame); fig.savefig(out_png, facecolor=bg)
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    ani.save(out_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=6000), savefig_kwargs=dict(facecolor=bg))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--mu-dry", type=float, default=0.7)
    ap.add_argument("--mu-wet", type=float, default=0.55)
    ap.add_argument("--wet-len", type=float, default=25.0, help="젖은 구간 길이 [m]")
    ap.add_argument("--before", type=float, default=8.0, help="젖은 구간 진입 몇 초 전부터 보여줄지")
    ap.add_argument("--after", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--k-lat", type=float, default=0.75, help="속도 계획에 쓰는 횡가속도 여유(μg 대비)")
    ap.add_argument("--no-render", action="store_true", help="숫자만 확인")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    tr = load_track(args.track)
    v_ref = speed_profile(tr, args.mu_dry, k_lat=args.k_lat)
    # 젖은 구간: 횡가속도 요구(v^2*κ)가 가장 큰 코너의 입구부터
    demand = v_ref ** 2 * np.abs(tr["kappa"])
    c = int(np.argmax(demand)); s_c = tr["s"][c]
    wet = (max(0.0, s_c - 0.4 * args.wet_len), s_c + 0.6 * args.wet_len)
    mu_of_s = lambda s: args.mu_wet if wet[0] <= s <= wet[1] else args.mu_dry
    lap_time = np.cumsum(np.r_[0, np.diff(tr["s"]) / np.maximum(v_ref[:-1], 0.3)])  # 주행선 위치 s까지 걸리는 시간
    t_wet_lap = float(np.interp(wet[0], tr["s"], lap_time))
    t_start = max(0.0, t_wet_lap - args.before - 3.0)            # 젖은 구간 몇 초 전에서 출발
    start = int(np.searchsorted(lap_time, t_start))
    t_wet = t_wet_lap - t_start
    duration = t_wet + args.after + 2.0
    print(f"트랙 {tr['length']:.0f} m, 젖은 구간 s={wet[0]:.0f}~{wet[1]:.0f} m (예상 진입 {t_wet:.1f} s), 시뮬레이션 {duration:.0f} s")

    t0 = time.time(); sim = simulate(tr, v_ref, mu_of_s, duration, start=start); print(f"시뮬레이션 {time.time()-t0:.0f}s")
    dev = np.min(np.linalg.norm(np.stack([sim["X"], sim["Y"]], 1)[:, None] - tr["center"][None], axis=2), axis=1)
    print(f"최대 차체 슬립 {np.degrees(np.abs(np.arctan2(sim['uy'], sim['ux']))).max():.1f}°, 최대 뒷바퀴 슬립 {np.degrees(np.abs(sim['alpha_r'])).max():.1f}°, "
          f"중심선에서 최대 {dev.max():.2f} m (트랙 반폭 1.1 m) -> {'트랙 안' if dev.max() < 1.1 else '트랙 이탈'}")

    step = int(round(100 / args.fps))
    s_in = np.argmax(sim["mu"] < args.mu_dry) if (sim["mu"] < args.mu_dry).any() else len(sim["t"]) // 2
    frames = np.arange(max(4, s_in - int(args.before * 100)), min(len(sim["t"]) - 101, s_in + int(args.after * 100)), step)
    paths = {}
    for p in args.ckpt:
        model, ck = load_checkpoint(p)
        paths[ck["kind"]] = predict_paths(model, sim, frames)
    truth = truth_paths(sim, frames)
    for k in paths:
        e = np.linalg.norm(paths[k][:, -1] - truth[:, -1], axis=1)
        print(f"{k}: 1초 뒤 위치 오차 평균 {e.mean():.2f} m, 최대 {e.max():.2f} m")

    if args.no_render:
        return
    os.makedirs(os.path.join(ROOT, "viz", "out"), exist_ok=True); os.makedirs(os.path.join(ROOT, "docs", "figures"), exist_ok=True)
    out_mp4 = os.path.join(ROOT, "viz", "out", "demo_track.mp4"); out_png = os.path.join(ROOT, "docs", "figures", "demo_track_frame.png")
    png_fi = int(np.argmax(np.linalg.norm(paths.get("m0", next(iter(paths.values())))[:, -1] - truth[:, -1], axis=1)))
    t0 = time.time(); render(tr, sim, frames, paths, truth, wet, out_mp4, out_png, args.fps, png_frame=png_fi)
    print(f"렌더링 {time.time()-t0:.0f}s -> {out_mp4}, {out_png}")


if __name__ == "__main__":
    main()
