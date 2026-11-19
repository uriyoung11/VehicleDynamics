"""우리 물리 모델(P0~P2)을 CommonRoad vehicle models 3.0.2와 같은 입력으로 비교한다.

  python -m physics.commonroad_check            # 표 출력 + docs/figures/commonroad_check.png

비교 쌍
  P0 ~ ST  : CommonRoad ST는 작은 각도로 선형화한 모델이고 속도를 입력(v' = a)으로 받는다.
             그래서 궤적이 아니라 같은 상태에서의 미분값(r', β')을 비교하고,
             각도가 작아질수록 차이가 각도²에 비례해 0으로 가는지 본다.
  P1 ~ STD : STD(단일 트랙 드리프트)는 Pacejka 타이어 + 바퀴 회전 상태를 가진다.
             캠버 0에서 STD의 횡방향 Pacejka는 우리 pacejka_lat와 같은 식이므로
             계수를 그대로 옮기고, 가감속이 없는 시나리오에서 궤적을 비교한다.
  P2 ~ STD : 가감속이 있는 시나리오. 하중 이동은 같은 식이지만 결합 슬립은
             우리 마찰 타원 vs STD의 Pacejka 결합 슬립 식으로 다르다(의도된 차이).

차량: CommonRoad vehicle 2 (BMW 320i). 조향 각속도·가속 제한은 풀어서 입력을 그대로 넣는다.
"""
import math, os, copy
import numpy as np
import torch

from .params import VehicleParams, TireParams
from .bicycle import derivatives, ModelOptions
from .tires import pacejka_lat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

P0_OPT = ModelOptions(tire="linear", load_transfer=False, combined=False, resistance=False)
P1_OPT = ModelOptions(tire="pacejka", load_transfer=False, combined=False, resistance=False)
P2_OPT = ModelOptions(tire="pacejka", load_transfer=True, combined=True, resistance=False)        # 마찰 타원
P2D_OPT = ModelOptions(tire="pacejka", load_transfer=True, combined="derate", resistance=False)  # 피크만 줄임


def cr_params():
    from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
    p = parameters_vehicle2()
    p.steering.v_max, p.steering.v_min = 100.0, -100.0
    p.longitudinal.a_max, p.longitudinal.v_switch = 50.0, 1e3
    return p


def our_params(p):
    """CommonRoad 파라미터를 우리 VehicleParams로 옮긴다. 횡방향 Pacejka 계수는 그대로."""
    t = p.tire
    tire = TireParams(c_alpha=-t.p_ky1, C=t.p_cy1, E=t.p_ey1)
    vp = VehicleParams(
        name="cr_bmw320i", m=p.m, Iz=p.I_z, a=p.a, b=p.b, h=p.h_s,
        tire_f=tire, tire_r=tire, delta_max=p.steering.max, delta_rate_max=100.0,
        drive_front=p.T_se, brake_front=p.T_sb, power_max=1e9)
    return vp, t.p_dy1   # μ = p_dy1


# ---------------------------------------------------------------- 입력 시나리오

def smooth(t, t0, T):
    """0→1 부드러운 계단(3x²-2x³)과 그 미분."""
    x = np.clip((t - t0) / T, 0.0, 1.0)
    inside = (t > t0) & (t < t0 + T)
    return 3 * x ** 2 - 2 * x ** 3, np.where(inside, 6 * x * (1 - x) / T, 0.0)


def scenario(name):
    """반환: v0, 지속시간, f(t) -> (delta, delta_rate, ax)"""
    d = math.radians
    if name == "step_small":        # 선형 영역
        def f(t):
            s, ds = smooth(t, 0.5, 0.3); return d(1.0) * s, d(1.0) * ds, 0.0
        return 20.0, 4.0, f
    if name == "step_limit":        # 앞바퀴 포화까지
        def f(t):
            s, ds = smooth(t, 0.5, 0.3); return d(10.0) * s, d(10.0) * ds, 0.0
        return 20.0, 4.0, f
    if name == "sine_limit":        # 좌우 반복, 한계 근처
        def f(t):
            w = 2 * math.pi * 0.5; on = t >= 0.5
            return (d(8.0) * math.sin(w * (t - 0.5)) if on else 0.0,
                    d(8.0) * w * math.cos(w * (t - 0.5)) if on else 0.0, 0.0)
        return 20.0, 5.0, f
    if name == "brake_mild":        # 선회 중 약한 제동: 하중 이동으로 뒤가 서서히 빠진다
        def f(t):
            s, ds = smooth(t, 0.5, 0.3); b, _ = smooth(t, 2.0, 0.3)
            return d(2.5) * s, d(2.5) * ds, -3.0 * float(b)
        return 20.0, 4.0, f
    if name == "brake_in_turn":     # 선회 중 강한 제동: 뒤가 빠져 스핀
        def f(t):
            s, ds = smooth(t, 0.5, 0.3); b, _ = smooth(t, 2.0, 0.3)
            return d(2.5) * s, d(2.5) * ds, -6.0 * float(b)
        return 20.0, 4.0, f
    if name == "accel_in_turn":     # 선회 중 가속(후륜구동): 뒷바퀴 결합 슬립
        def f(t):
            s, ds = smooth(t, 0.5, 0.3); b, _ = smooth(t, 2.0, 0.3)
            return d(6.0) * s, d(6.0) * ds, 4.0 * float(b)
        return 12.0, 4.0, f
    raise KeyError(name)


SCENARIOS = {  # 시나리오 -> 비교할 우리 모델
    "step_small": ("P1", P1_OPT), "step_limit": ("P1", P1_OPT), "sine_limit": ("P1", P1_OPT),
    "brake_mild": ("P2", P2_OPT), "brake_in_turn": ("P2", P2_OPT), "accel_in_turn": ("P2", P2_OPT),
}


# ---------------------------------------------------------------- 적분

def rk4(fun, x, t, dt):
    k1 = fun(t, x); k2 = fun(t + dt / 2, x + dt / 2 * k1)
    k3 = fun(t + dt / 2, x + dt / 2 * k2); k4 = fun(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run_std(p, v0, dur, f, dt):
    from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std
    from vehiclemodels.init_std import init_std
    x = np.array(init_std([0.0, 0.0, 0.0, v0, 0.0, 0.0, 0.0], p), dtype=float)

    def fun(t, x):
        _, dd, ax = f(t)
        return np.array(vehicle_dynamics_std(list(x), [float(dd), float(ax)], p))

    n = int(round(dur / dt)); out = np.zeros((n + 1, 6))
    for i in range(n + 1):
        t = i * dt
        v, beta = x[3], x[6]
        out[i] = [x[0], x[1], x[5], v * math.cos(beta), v * math.sin(beta), x[2]]   # X Y r ux uy delta
        if i < n:
            x = rk4(fun, x, t, dt)
    return out


def run_ours(vp, mu, opt, v0, dur, f, dt):
    x = torch.tensor([0.0, 0.0, 0.0, v0, 0.0, 0.0], dtype=torch.float64)
    m = vp.m

    def fun(t, x):
        de, _, ax = f(t)
        u = torch.tensor([float(de), m * float(ax)], dtype=torch.float64)
        return derivatives(x, u, vp, mu, opt)

    n = int(round(dur / dt)); out = np.zeros((n + 1, 6)); amax = np.zeros(2)
    for i in range(n + 1):
        t = i * dt
        de = float(f(t)[0])
        out[i] = [x[0], x[1], x[5], x[3], x[4], de]
        if i < n:
            x = rk4(fun, x, t, dt)
    return out


def alpha_ratio(vp, mu, traj):
    """궤적에서 |α|/α_peak 최댓값 (앞, 뒤). 포화에 들어갔는지 확인용."""
    C = vp.tire_f.C
    B = vp.tire_f.c_alpha / (C * mu)
    # E≈0이므로 C·atan(Bα)=π/2 지점을 근사 peak로 쓴다
    a_peak = math.tan(math.pi / (2 * C)) / B
    ux, uy, r, de = traj[:, 3], traj[:, 4], traj[:, 2], traj[:, 5]
    af = np.arctan((uy + vp.a * r) / ux) - de
    ar = np.arctan((uy - vp.b * r) / ux)
    return np.abs(af).max() / a_peak, np.abs(ar).max() / a_peak


# ---------------------------------------------------------------- P0 ~ ST 미분값 비교

def compare_p0_st(p, vp, mu, n=2000, seed=0):
    from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
    rng = np.random.default_rng(seed)
    rows = []
    for eps in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        errs = []
        for _ in range(n // 10):
            v = rng.uniform(10, 30)
            beta, de = rng.uniform(-eps, eps, 2)
            r = rng.uniform(-eps, eps) * v / p.a
            fst = vehicle_dynamics_st([0, 0, de, v, 0, r, beta], [0.0, 0.0], p)
            x = torch.tensor([0, 0, 0, v * math.cos(beta), v * math.sin(beta), r], dtype=torch.float64)
            xd = derivatives(x, torch.tensor([de, 0.0], dtype=torch.float64), vp, mu, P0_OPT).numpy()
            ux, uy = x[3].item(), x[4].item()
            beta_dot = (ux * xd[4] - uy * xd[3]) / v ** 2
            errs.append([xd[5] - fst[5], beta_dot - fst[6], abs(fst[5]), abs(fst[6])])
        e = np.array(errs)
        rows.append(dict(eps=eps, rel_r=np.abs(e[:, 0]).max() / e[:, 2].max(),
                         rel_beta=np.abs(e[:, 1]).max() / e[:, 3].max()))
    return rows


def compare_p1_std(p, vp, mu, n=200, seed=0, amps=(0.01, 0.05, 0.15)):
    """같은 상태에서 P1과 STD의 미분값(r', β') 비교. STD 바퀴 속도는 종방향 힘이 0이 되는 값으로 둔다
    (가감속 없을 때 바퀴가 자연스럽게 가는 상태)."""
    from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std
    import vehiclemodels.utils.tire_model as tm

    def slip_zero_force(fz):
        lo, hi = -0.05, 0.05
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tm.formula_longitudinal(mid, 0, fz, p.tire) > 0 else (lo, mid)
        return mid

    L = p.a + p.b
    sf, sr = slip_zero_force(p.m * 9.81 * p.b / L), slip_zero_force(p.m * 9.81 * p.a / L)
    rng = np.random.default_rng(seed); rows = []
    for amp in amps:
        e = []
        for _ in range(n):
            v = rng.uniform(10, 30); beta, de = rng.uniform(-amp, amp, 2); r = rng.uniform(-amp, amp) * v / p.a
            uwf = v * math.cos(beta) * math.cos(de) + (v * math.sin(beta) + p.a * r) * math.sin(de)
            uwr = v * math.cos(beta)
            f = vehicle_dynamics_std([0, 0, de, v, 0, r, beta, (1 - sf) * uwf / p.R_w, (1 - sr) * uwr / p.R_w], [0.0, 0.0], p)
            x = torch.tensor([0, 0, 0, v * math.cos(beta), v * math.sin(beta), r], dtype=torch.float64)
            xd = derivatives(x, torch.tensor([de, 0.0], dtype=torch.float64), vp, mu, P1_OPT).numpy()
            bd = (x[3].item() * xd[4] - x[4].item() * xd[3]) / v ** 2
            e.append([xd[5] - f[5], bd - f[6], f[5], f[6]])
        e = np.array(e)
        rows.append(dict(eps=amp, rel_r=np.abs(e[:, 0]).max() / np.abs(e[:, 2]).max(),
                         rel_beta=np.abs(e[:, 1]).max() / np.abs(e[:, 3]).max()))
    return rows


# ---------------------------------------------------------------- 실행

def metrics(ref, our):
    """r, β, 속도: 최대 오차 / 기준 신호의 최대 크기. 위치: 끝점 거리 [m]."""
    beta_ref = np.arctan2(ref[:, 4], ref[:, 3]); beta_our = np.arctan2(our[:, 4], our[:, 3])
    v_ref = np.hypot(ref[:, 3], ref[:, 4]); v_our = np.hypot(our[:, 3], our[:, 4])
    rel = lambda a, b: np.abs(a - b).max() / max(np.abs(a).max(), 1e-9)
    return dict(r=rel(ref[:, 2], our[:, 2]), beta=rel(beta_ref, beta_our),
                v=np.abs(v_ref - v_our).max() / v_ref.max(),
                pos=float(np.hypot(*(ref[-1, :2] - our[-1, :2]))),
                dist=float(np.sum(np.hypot(*np.diff(ref[:, :2], axis=0).T))))


def run_all(dt=5e-4):
    p = cr_params(); vp, mu = our_params(p)
    res = {"p0_st": compare_p0_st(p, vp, mu), "p1_std": compare_p1_std(p, vp, mu), "traj": {}}
    for name, (label, opt) in SCENARIOS.items():
        v0, dur, f = scenario(name)
        ref = run_std(p, v0, dur, f, dt)
        our = run_ours(vp, mu, opt, v0, dur, f, dt)
        extra = {}
        if label == "P2":   # 결합 방식 대안(derate)과 P1(하중 이동·결합 없음)도 같이 돌린다
            extra["P2-derate"] = run_ours(vp, mu, P2D_OPT, v0, dur, f, dt)
            extra["P1"] = run_ours(vp, mu, P1_OPT, v0, dur, f, dt)
        res["traj"][name] = dict(label=label, t=np.arange(len(ref)) * dt, ref=ref, our=our, extra=extra,
                                 m=metrics(ref, our), sat=alpha_ratio(vp, mu, ref),
                                 m_extra={k: metrics(ref, v) for k, v in extra.items()})
    return res


def print_tables(res):
    print("\n[P0 ~ ST] 같은 상태에서 미분값 상대 오차 (최대 오차 / 최대 크기)")
    print(f"{'각도 범위':>10} {'r_dot':>10} {'beta_dot':>10}")
    for row in res["p0_st"]:
        print(f"{row['eps']:>8.3f}rad {row['rel_r']:>10.2e} {row['rel_beta']:>10.2e}")
    print("\n[P1 ~ STD] 같은 상태에서 미분값 상대 오차")
    for row in res["p1_std"]:
        print(f"{row['eps']:>8.3f}rad {row['rel_r']:>10.2e} {row['rel_beta']:>10.2e}")
    print("\n[P1/P2 ~ STD] 궤적 비교 (r, β, 속도: 최대 오차 / 최대 크기, 위치: 끝점 거리)")
    print(f"{'시나리오':<15}{'모델':<10}{'|α|/α_peak 앞,뒤':>18}{'r':>9}{'β':>9}{'v':>9}{'위치[m]':>9}{'주행[m]':>8}")
    for name, d in res["traj"].items():
        for lab, m in [(d["label"], d["m"])] + list(d["m_extra"].items()):
            print(f"{name:<15}{lab:<10}{d['sat'][0]:>9.2f},{d['sat'][1]:>5.2f}   "
                  f"{m['r']:>9.4f}{m['beta']:>9.4f}{m['v']:>9.4f}{m['pos']:>9.3f}{m['dist']:>8.1f}")


def plot(res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    REF, OUR, P1C = "#2a78d6", "#eb6834", "#8a8a85"
    EXTRA = {"P2-derate": dict(color="#1baf7a", lw=2, ls="-.", label="우리 P2 (피크만 줄임)"),
             "P1": dict(color=P1C, lw=1.2, ls=":", label="우리 P1 (하중 이동·결합 없음)")}
    names = list(res["traj"])
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 2.3 * len(names)), constrained_layout=True)
    titles = {"step_small": "계단 조향 1° (선형)", "step_limit": "계단 조향 10° (앞바퀴 포화)",
              "sine_limit": "사인 조향 ±8° (한계 근처)", "brake_mild": "선회 중 제동 −3 m/s² (뒤가 서서히 빠짐)", "brake_in_turn": "선회 중 제동 −6 m/s² (스핀)",
              "accel_in_turn": "선회 중 가속 +4 m/s² (후륜구동)"}
    for i, name in enumerate(names):
        d = res["traj"][name]; t, ref, our = d["t"], d["ref"], d["our"]
        beta = lambda z: np.degrees(np.arctan2(z[:, 4], z[:, 3]))
        series = [("요레이트 r [rad/s]", lambda z: z[:, 2]), ("차체 슬립각 β [°]", beta),
                  ("속도 [m/s]", lambda z: np.hypot(z[:, 3], z[:, 4]))]
        for j, (ylab, g) in enumerate(series):
            ax = axes[i, j]
            ax.plot(t, g(ref), color=REF, lw=2, label="CommonRoad STD")
            ax.plot(t, g(our), color=OUR, lw=2, ls="--",
                    label="우리 P2 (마찰 타원)" if d["label"] == "P2" else f"우리 {d['label']}")
            for k, z in d["extra"].items():
                ax.plot(t, g(z), **EXTRA[k])
            ax.grid(alpha=0.25, lw=0.6)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            if j == 0: ax.set_title(titles[name], loc="left", fontsize=11, fontweight="bold")
            if i == len(names) - 1: ax.set_xlabel("시간 [s]")
            ax.set_ylabel(ylab, fontsize=9)
            if j == 2 and i in (0, 3): ax.legend(fontsize=8, frameon=False, loc="lower left")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130); print("saved", path)


if __name__ == "__main__":
    torch.set_num_threads(1)
    res = run_all()
    print_tables(res)
    plot(res, os.path.join(ROOT, "docs", "figures", "commonroad_check.png"))
