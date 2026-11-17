"""시각화 장면(Spielberg, 젖은 구간)을 여러 제어기로 다시 주행해 비교한다. MPC 도입 1·2단계.

  python -m control.run_scene                               # pure pursuit, MPPI(마른 μ), MPPI(정답 μ)
  python -m control.run_scene --ctrl pp mppi_dry mppi_m0 mppi_b1 --ckpt-dir checkpoints/f1tenth_v3_trial

제어기
  pp          지금 시각화의 pure pursuit + 속도표 P 제어 (100 Hz)
  mppi_dry    MPPI + P2 물리, μ는 마른 노면 값만 안다 (현실적인 조건)
  mppi_oracle MPPI + P2 물리, 젖은 구간 μ까지 안다 (상한선)
  mppi_<모델>  MPPI + 학습 모델(m0, b1, m1, m2, m3). Ux는 공통 물리식

실제 차: P2(하중 이동, 결합 슬립), 1 ms RK4, 조향 각속도 제한. 젖은 구간 μ 0.55, 나머지 0.70.
출력: viz/out/mpc/<제어기>[_tag].npz, docs/figures/mpc_scene[_tag].png, 화면 표. 기존 시각화 결과는 건드리지 않는다.
"""
import argparse, os, time
import numpy as np
import torch

from physics import F1TENTH, P2, rk4_step
from viz.demo_track import load_track, speed_profile
from control.mppi import MPPI, MPPIConfig, TrackCost, PhysicsDynamics, LearnedDynamics, G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "viz", "out", "mpc")
P = F1TENTH


def scenario(args):
    """viz/demo_rerun.scenario와 같은 장면(가장 빠른 코너 근처 젖은 구간, 그 앞뒤 구간)."""
    tr = load_track(args.track)
    # 장면 배치(젖은 구간 위치, 출발점, 주행 시간)는 기본 설정(k 0.75, v_max 7)으로 고정한다.
    # 여유 계수나 속도 상한을 바꿔도 같은 장면에서 목표 속도만 달라지게 하려는 것이다.
    v_base = speed_profile(tr, args.mu_dry, k_lat=0.75, v_max=7.0)
    v_ref = speed_profile(tr, args.mu_dry, k_lat=args.k_lat, v_max=getattr(args, "v_max", 7.0))
    c = int(np.argmax(v_base ** 2 * np.abs(tr["kappa"]))); s_c = tr["s"][c]
    wet = (max(0.0, s_c - 0.4 * args.wet_len), s_c + 0.6 * args.wet_len)
    lap_time = np.cumsum(np.r_[0, np.diff(tr["s"]) / np.maximum(v_base[:-1], 0.3)])
    t_wet = float(np.interp(wet[0], tr["s"], lap_time))
    t_start = max(0.0, t_wet - args.before - 3.0)
    start = int(np.searchsorted(lap_time, t_start))
    duration = (t_wet - t_start) + args.after + 2.0
    return tr, v_ref, wet, start, duration


def pure_pursuit(X, Y, psi, ux, xy, idx, p, v_ref, ld0=1.0, ld_k=0.5):
    n = len(xy)
    Ld = ld0 + ld_k * ux
    j = idx
    while np.hypot(*(xy[j] - [X, Y])) < Ld:
        j = (j + 1) % n
    dxl = (xy[j, 0] - X) * np.cos(psi) + (xy[j, 1] - Y) * np.sin(psi)
    dyl = -(xy[j, 0] - X) * np.sin(psi) + (xy[j, 1] - Y) * np.cos(psi)
    d_cmd = np.clip(np.arctan2(2 * p.L * dyl, dxl ** 2 + dyl ** 2), -p.delta_max, p.delta_max)
    fx = p.m * 2.0 * (v_ref[idx] - ux) + (p.c0 + p.c1 * ux + p.c2 * ux ** 2)
    return d_cmd, fx


def run(name, args, tr, v_ref, wet, start, duration, models):
    xy, n = tr["xy"], len(tr["xy"])
    mu_at = lambda i: args.mu_wet if wet[0] <= tr["s"][i] <= wet[1] else args.mu_dry
    psi0 = np.arctan2(*(xy[(start + 1) % n] - xy[start])[::-1])
    x = torch.tensor([[xy[start, 0], xy[start, 1], psi0, v_ref[start], 0.0, 0.0]], dtype=torch.float64)
    delta, d_cmd, fx_cmd, idx = 0.0, 0.0, 0.0, start
    long_scale = P.m * 100
    hist = [[0.0, 0.0, float(v_ref[start]), 0.0, 0.0]] * 4

    ctrl = None
    if name.startswith("mppi"):
        cfg = MPPIConfig(K=args.K, H=args.H, v_max=args.v_max)
        cost = TrackCost(tr, v_ref, cfg, mode="progress" if args.mode == "B" else "track")
        kind = name[5:]
        s_t, wet_t = torch.as_tensor(tr["s"]), wet

        def mu_true(q):   # 정답 μ 지도 (위치 → μ)
            j = cost.nearest(q, mppi_idx[0])
            s = s_t[j]
            return torch.where((s >= wet_t[0]) & (s <= wet_t[1]), args.mu_wet, args.mu_dry).to(torch.float64)
        if kind in ("dry", "oracle"):
            if kind == "dry":
                mu_fn = lambda q: torch.full(q.shape[:1], args.mu_dry, dtype=torch.float64)
            else:
                mu_fn = mu_true
            dyn = PhysicsDynamics(P, P2, mu_fn)
        else:
            dyn = LearnedDynamics(models[kind], P, long_scale, (P.drive_front, P.brake_front),
                                  mu_fn=mu_true if args.long_mu_oracle else None)
        ctrl = MPPI(dyn, cost, P, cfg)
        mppi_idx = [start]

    rec, solve, plans = [], [], []
    dt_sim, dt_log = 0.001, 0.01
    n_ctrl = int(round((args.dt_mpc if ctrl else 0.01) / dt_sim))
    n_log = int(round(dt_log / dt_sim))
    steps = int(duration / dt_sim)
    for k in range(steps):
        X, Y, psi, ux, uy, r = (float(v) for v in x[0])
        if k % n_log == 0:
            win = (idx + np.arange(-5, 60)) % n
            d2 = np.sum((xy[win] - [X, Y]) ** 2, 1)
            idx = int(win[np.argmin(d2)]) if d2.min() < 4.0 else int(np.argmin(np.sum((xy - [X, Y]) ** 2, 1)))
        if k % n_ctrl == 0:
            if ctrl is None:
                d_cmd, fx_cmd = pure_pursuit(X, Y, psi, ux, xy, idx, P, v_ref)
            else:
                mppi_idx[0] = idx
                t0 = time.time()
                state = dict(x=x[0].numpy(), delta=delta, hist=np.array(hist)[None], mu_lim=args.mu_dry)
                u, info = ctrl.step(state, idx, args.mu_dry)
                solve.append(time.time() - t0)
                d_cmd, fx_cmd = float(u[0]), float(u[1]) * P.m * G
                if len(plans) < 100000:
                    plans.append((k * dt_sim, info["plan"]))
        mu = mu_at(idx)
        fx = float(np.clip(fx_cmd, -mu * P.m * G, min(P.power_max / max(ux, 0.5), mu * P.m * G)))
        delta += float(np.clip(d_cmd - delta, -P.delta_rate_max * dt_sim, P.delta_rate_max * dt_sim))
        u_t = torch.tensor([[delta, fx]], dtype=torch.float64)
        if k % n_log == 0:
            fxf = P.drive_front * fx if fx > 0 else P.brake_front * fx
            hist = hist[1:] + [[r, uy, ux, delta, fxf / long_scale]]
            rec.append([k * dt_sim, X, Y, psi, ux, uy, r, delta, fx, mu, idx])
        x = rk4_step(x, u_t, P, torch.tensor([mu], dtype=torch.float64), dt_sim, P2)
        if not torch.isfinite(x).all() or ux < 0.3:
            print(f"  [{name}] 시뮬레이션 중단 (t={k * dt_sim:.2f}s)")
            break
    a = np.array(rec)
    cols = ["t", "X", "Y", "psi", "ux", "uy", "r", "delta", "fx", "mu", "idx"]
    out = {c: a[:, i] for i, c in enumerate(cols)}
    out["solve"] = np.array(solve)
    return out


def metrics(sim, tr, v_ref):
    from viz.demo_rerun import Margin
    m, j = Margin(tr)(np.c_[sim["X"], sim["Y"]])
    dt = np.diff(sim["t"], prepend=0)
    t = sim["t"]
    e = np.sum((np.c_[sim["X"], sim["Y"]] - tr["center"][j]) * np.stack([-np.gradient(tr["center"][:, 1]), np.gradient(tr["center"][:, 0])], 1)[j]
               / np.linalg.norm(np.gradient(tr["center"], axis=0)[j], axis=1, keepdims=True), 1)
    s = tr["s"][sim["idx"].astype(int)]
    prog = np.unwrap(s / tr["length"] * 2 * np.pi) / (2 * np.pi) * tr["length"]
    return dict(min_margin=float(m.min()), off_time=float(dt[m < 0].sum()), max_off=float(max(0.0, -m.min())),
                lat_rms=float(np.sqrt(np.mean(e ** 2))), v_mean=float(sim["ux"].mean()),
                progress=float(prog[-1] - prog[0]), duration=float(t[-1]),
                solve_ms=float(sim["solve"].mean() * 1000) if len(sim["solve"]) else 0.0), m


def plot(results, tr, wet, path, mu_wet):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    cols = {"pp": "#888888", "mppi_dry": "#1f77b4", "mppi_oracle": "#2ca02c", "mppi_m0": "#ff7f0e",
            "mppi_b1": "#17becf", "mppi_m1": "#9467bd", "mppi_m2": "#8c564b", "mppi_m3": "#e377c2"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    for k in ("left", "right"):
        ax.plot(*tr[k].T, color="k", lw=0.8)
    wi = (tr["s"] >= wet[0]) & (tr["s"] <= wet[1])
    ax.fill(np.r_[tr["left"][wi, 0], tr["right"][wi, 0][::-1]], np.r_[tr["left"][wi, 1], tr["right"][wi, 1][::-1]],
            color="#2a6cff", alpha=0.2, label=f"젖은 구간 (μ {mu_wet})")
    allxy = []
    for name, (sim, met, m) in results.items():
        ax.plot(sim["X"], sim["Y"], color=cols.get(name, None), lw=1.4, label=name)
        bx.plot(sim["t"], m, color=cols.get(name, None), lw=1.4, label=f"{name} (최소 {met['min_margin']:+.2f} m)")
        allxy.append(np.c_[sim["X"], sim["Y"]][np.abs(sim["t"] - sim["t"][np.argmin(m)]) < 2.5])
    z = np.vstack(allxy)
    ax.set_xlim(z[:, 0].min() - 3, z[:, 0].max() + 3); ax.set_ylim(z[:, 1].min() - 3, z[:, 1].max() + 3)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="best"); ax.set_title("이탈 지점 근처 주행 궤적")
    bx.axhline(0, color="r", lw=0.8, ls="--"); bx.set_xlabel("시간 [s]"); bx.set_ylabel("트랙 가장자리까지 여유 [m]")
    bx.legend(fontsize=8); bx.set_title("가장자리까지 여유 (0 아래 = 이탈)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


COLORS = {"pp": "#888888", "mppi_dry": "#1f77b4", "mppi_oracle": "#2ca02c", "mppi_m0": "#ff7f0e",
          "mppi_b1": "#17becf", "mppi_m1": "#9467bd", "mppi_m2": "#8c564b", "mppi_m3": "#e377c2"}
LABEL = {"pp": "pure pursuit", "mppi_dry": "물리, 마른 μ만 앎", "mppi_oracle": "물리, 정답 μ (상한선)",
         "mppi_m0": "M0", "mppi_b1": "B1", "mppi_m1": "M1", "mppi_m2": "M2", "mppi_m3": "M3"}


def end_reason(sim, duration):
    """주행이 끝난 이유. 속도 0.3 m/s 미만이면 시뮬레이션을 멈추는데, 그것이 스핀인지 그냥 멈춘 것인지 구분한다."""
    if sim["t"][-1] >= duration - 0.05:
        return ""
    n = len(sim["t"])
    beta = np.degrees(np.arctan2(sim["uy"][-100:], np.maximum(np.abs(sim["ux"][-100:]), 0.1)))
    return "스핀" if np.abs(beta).max() > 30 else "멈춤"


def failure(sim, tr, duration):
    """실패 시점과 이유. 경주 기준으로 트랙 경계를 처음 넘은 순간(차 중심 여유 < 0)을 실패로 본다.
    시뮬레이터에는 벽이 없어 이탈해도 계속 달리므로, 그 뒤 주행은 분석용으로만 남긴다.
    반환: (실패한 기록 번호 또는 None, "이탈" | "스핀" | "멈춤" | "")"""
    from viz.demo_rerun import Margin
    m, _ = Margin(tr)(np.c_[sim["X"], sim["Y"]])
    off = np.flatnonzero(m < 0)
    if len(off):
        return int(off[0]), "이탈"
    why = end_reason(sim, duration)
    return (len(sim["t"]) - 1, why) if why else (None, "")


def track_progress(sim, tr):
    """출발점부터 중심선을 따라 간 거리 [m]."""
    s = tr["s"][sim["idx"].astype(int)]
    p = np.unwrap(s / tr["length"] * 2 * np.pi) / (2 * np.pi) * tr["length"]
    return p - p[0]


def plot_compare(results, tr, wet, path, mu_wet, title):
    """트랙 위 궤적 + 트랙 거리별 속도 + 트랙 거리별 가장자리 여유. 제어기를 같은 위치끼리 비교한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
    ax, bx, cx = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for k in ("left", "right"):
        ax.plot(*tr[k].T, color="k", lw=0.8)
    wi = (tr["s"] >= wet[0]) & (tr["s"] <= wet[1])
    ax.fill(np.r_[tr["left"][wi, 0], tr["right"][wi, 0][::-1]], np.r_[tr["left"][wi, 1], tr["right"][wi, 1][::-1]],
            color="#2a6cff", alpha=0.2, label=f"젖은 구간 (μ {mu_wet})")
    s0 = None
    allxy = []
    for name, (sim, met, m) in results.items():
        col, lab = COLORS.get(name), LABEL.get(name, name)
        p = track_progress(sim, tr)
        if s0 is None:
            s0 = tr["s"][int(sim["idx"][0])]
        fi, why = failure(sim, tr, results_duration(results))
        end = f" ({why} t={sim['t'][fi]:.1f}s)" if why else " (완주)"
        ax.plot(sim["X"], sim["Y"], color=col, lw=1.3, label=lab + end)
        if fi is not None:
            ax.plot(sim["X"][fi], sim["Y"][fi], "x", color=col, ms=11, mew=2.5)
        bx.plot(p, sim["ux"], color=col, lw=1.3, label=f"{lab}: 평균 {met['v_mean']:.2f} m/s{end}")
        cx.plot(p, m, color=col, lw=1.3, label=f"{lab}: 최소 {met['min_margin']:+.2f} m, 이탈 {met['off_time']:.2f}s")
        allxy.append(np.c_[sim["X"], sim["Y"]])
    z = np.vstack(allxy)
    ax.set_xlim(z[:, 0].min() - 2, z[:, 0].max() + 2); ax.set_ylim(z[:, 1].min() - 2, z[:, 1].max() + 2)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="best"); ax.set_title("주행 궤적")
    w0, w1 = (wet[0] - s0) % tr["length"], (wet[1] - s0) % tr["length"]
    for a in (bx, cx):
        a.axvspan(w0, w1, color="#2a6cff", alpha=0.12, lw=0)
        a.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    bx.set_ylabel("속도 Ux [m/s]"); bx.legend(fontsize=8, loc="lower left"); bx.set_title("트랙 위치별 속도 (파란 배경 = 젖은 구간)")
    cx.axhline(0, color="r", lw=0.8, ls="--"); cx.set_ylabel("가장자리까지 여유 [m]"); cx.set_xlabel("출발점부터 트랙 거리 [m]")
    cx.legend(fontsize=8, loc="lower left"); cx.set_title("트랙 위치별 가장자리 여유 (0 아래 = 이탈)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def results_duration(results):
    return max(met["duration"] for _, met, _ in results.values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", nargs="+", default=["pp", "mppi_dry", "mppi_oracle"])
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints", "f1tenth_v3_trial"))
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--mu-dry", type=float, default=0.7)
    ap.add_argument("--mu-wet", type=float, default=0.55)
    ap.add_argument("--k-lat", type=float, default=0.75)
    ap.add_argument("--v-max", type=float, default=7.0, help="속도 상한 [m/s]. 방식 A 목표 속도와 방식 B 상한에 같이 쓴다")
    ap.add_argument("--wet-len", type=float, default=25.0)
    ap.add_argument("--before", type=float, default=8.0)
    ap.add_argument("--after", type=float, default=10.0)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--dt-mpc", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="", help="출력 이름 접미사. 장면마다 결과를 따로 남긴다 (예: hard)")
    ap.add_argument("--mode", choices=["A", "B"], default=None,
                    help="A: 목표 속도 추종, B: 진행 최대화. 주면 출력 이름에 _A/_B가 붙고 새 비교 그림(mpc_<mode>...)을 만든다")
    ap.add_argument("--reuse", action="store_true", help="저장된 npz가 있으면 다시 돌리지 않고 읽는다")
    ap.add_argument("--long-mu-oracle", action="store_true",
                    help="진단용: 학습 모델의 종방향 힘 한계(제동·가속)만 정답 μ 지도로 자른다. 횡방향 예측은 그대로 모델이 한다")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "B" and "pp" in args.ctrl:
        print("방식 B에는 pure pursuit가 없다(목표 속도를 쓰므로). pp는 건너뛴다.")
        args.ctrl = [c for c in args.ctrl if c != "pp"]
    torch.set_num_threads(args.threads)
    tr, v_ref, wet, start, duration = scenario(args)
    models = {}
    from models.baselines import load_checkpoint
    for c in args.ctrl:
        if c.startswith("mppi_") and c[5:] not in ("dry", "oracle"):
            models[c[5:]], _ = load_checkpoint(os.path.join(args.ckpt_dir, f"{c[5:]}_s0.pt"))
            models[c[5:]] = models[c[5:]].double().eval()   # MPPI 래퍼는 float64로 굴린다
    os.makedirs(OUT, exist_ok=True)
    results = {}
    suffix = "".join("_" + x for x in (args.mode, args.tag) if x)
    for c in args.ctrl:
        t0 = time.time()
        f = os.path.join(OUT, f"{c}{suffix}.npz")
        if args.reuse and os.path.exists(f):     # 제어기별로 따로 돌린 결과를 모아 그림만 그릴 때
            sim = dict(np.load(f))
        else:
            sim = run(c, args, tr, v_ref, wet, start, duration, models)
            np.savez(f, **sim)
        met, m = metrics(sim, tr, v_ref)
        results[c] = (sim, met, m)
        print(f"{c:12s} 최소 여유 {met['min_margin']:+.3f} m | 이탈 {met['off_time']:.2f}s 최대 {met['max_off']:.3f} m | "
              f"횡오차 RMS {met['lat_rms']:.3f} m | 평균 속도 {met['v_mean']:.2f} m/s | 진행 {met['progress']:.1f} m / {met['duration']:.1f}s | "
              f"풀이 {met['solve_ms']:.0f} ms/회 | 계산 {time.time() - t0:.0f}s", flush=True)
    if args.no_plot:
        return
    if args.mode:
        fig = os.path.join(ROOT, "docs", "figures", f"mpc_{args.mode}{'_' + args.tag if args.tag else ''}.png")
        name = "방식 A: 목표 속도 추종" if args.mode == "A" else "방식 B: 진행 최대화 (목표 속도 없음)"
        plot_compare(results, tr, wet, fig, args.mu_wet,
                     f"{name} | {args.track}, 마른 μ {args.mu_dry}, 젖은 구간 μ {args.mu_wet}"
                     + (f", 목표 속도 = 한계 × {args.k_lat}" if args.mode == "A" else ""))
    else:
        fig = os.path.join(ROOT, "docs", "figures", f"mpc_scene{'_' + args.tag if args.tag else ''}.png")
        plot(results, tr, wet, fig, args.mu_wet)
    print("saved", fig)


if __name__ == "__main__":
    main()
