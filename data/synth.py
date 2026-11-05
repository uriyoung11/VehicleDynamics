"""D3 합성 데이터 생성기 (1단계: P2 자전거 모델 + 랜덤 입력 + μ 스윕).

여러 에피소드를 배치로 한 번에 적분한다. 각 에피소드는 하나의 연속 주행 궤적이다.

  python -m data.synth --vehicle f1tenth --episodes 200 --duration 20 --out data/generated/f1tenth_v1

입력 만들기 (100 Hz 제어 주기마다 갱신, 그 사이엔 고정):
  - 조향: 목표 횡가속도 k*μ*g*tanh(s(t))를 운동학 관계 δ = atan(L*a_y/Ux^2)로 바꾼 뒤,
          최대 조향각과 조향 각속도로 제한. s(t)는 OU 과정, k(공격성)는 에피소드마다 뽑는다.
          k가 1을 넘으면 의도적으로 한계를 넘는다.
  - 종방향: OU로 움직이는 목표 속도를 P 제어로 따라가고, 랜덤 가감속 성분을 더한다.
            랜덤 성분은 속도가 낮을수록 줄이고, 최저 속도 근처에서는 제동을 막는다.
            출력 한계(P/Ux)와 마찰 한계로 자른다.
  - 입력 모드(에피소드마다 하나):
      random : 위 그대로
      power  : 조향이 클수록 구동력을 더한다(선회 중 가속 → 뒷바퀴 포화, 파워 오버스티어)
      brake  : 조향이 클수록 제동력을 더한다(선회 중 제동)
    bang   : 일부 에피소드는 조향을 최대값에 붙이는 뱅뱅 조향으로 바꾼다(D2 실데이터의 조향 분포)

저장 형식 (에피소드당 npz 1개 + manifest.csv + config.json):
  시각 t_k의 상태와, [t_k, t_k+1) 동안 적용된 입력, 그 순간의 타이어 힘·슬립각·가속도.
  meas_* 는 잡음·편향이 들어간 측정값이다. 학습 입력은 meas_*, 진단·라벨은 참값을 쓴다.

1단계에서 빠진 것(다음 단계): 타이어 이완 지연, 4륜 모델, 트랙 주행·표준 기동 입력, 주행 중 μ 변화,
실데이터와 같은 필터 파이프라인.
"""
import argparse, csv, json, math, os, time
from dataclasses import asdict
import numpy as np
import torch

from physics import PRESETS, P2, rk4_step, derivatives, peak_slip

MU_SETS = {"train": [0.3, 0.5, 0.7, 1.0], "test_mu_interp": [0.4, 0.6, 0.85], "test_mu_extrap": [0.2]}
CHANNELS = ["t", "X", "Y", "psi", "ux", "uy", "r", "delta", "fx_cmd",
            "fxf", "fxr", "fyf", "fyr", "fzf", "fzr", "alpha_f", "alpha_r",
            "ux_dot", "uy_dot", "r_dot", "ax", "ay", "regime_ratio"]
MEAS = ["ux", "uy", "r", "ax", "ay"]


def assign_episodes(n, rng, frac_interp, frac_extrap, frac_val=0.15, frac_test=0.15):
    """에피소드별 (split, μ). μ holdout 그룹은 통째로 평가용, 학습 μ 그룹은 에피소드 단위로 나눈다."""
    out = []
    for _ in range(n):
        u = rng.random()
        if u < frac_extrap:
            out.append(("test_mu_extrap", rng.choice(MU_SETS["test_mu_extrap"])))
        elif u < frac_extrap + frac_interp:
            out.append(("test_mu_interp", rng.choice(MU_SETS["test_mu_interp"])))
        else:
            v = rng.random()
            split = "val" if v < frac_val else "test" if v < frac_val + frac_test else "train"
            out.append((split, rng.choice(MU_SETS["train"])))
    return out


class OU:
    """평균 0, 정상 표준편차 sigma, 시간상수 tau인 Ornstein-Uhlenbeck 과정."""
    def __init__(self, n, tau, sigma, dt, gen):
        self.a = math.exp(-dt / tau)
        self.s = sigma * math.sqrt(1 - self.a ** 2)
        self.gen = gen
        self.x = sigma * torch.randn(n, generator=gen, dtype=torch.float64)

    def step(self):
        self.x = self.a * self.x + self.s * torch.randn(self.x.shape, generator=self.gen, dtype=torch.float64)
        return self.x


MODES = ("random", "power", "brake")
BANG_FRAC_DEFAULT = {"f1tenth": 0.5, "fullscale": 0.0}


def simulate_batch(base, mus, k_aggr, mass_s, stiff_s, duration, dt_sim, dt_save, gen, opt=P2,
                   coupling=None, bang=None):
    """coupling: 에피소드별 조향-종방향 결합 계수(+ 선회 중 가속, - 선회 중 제동, 0 랜덤 모드).
    bang: 에피소드별 뱅뱅 조향 여부."""
    n = len(mus)
    mu = torch.tensor(mus, dtype=torch.float64)
    k = torch.tensor(k_aggr, dtype=torch.float64)
    cpl = torch.tensor(coupling if coupling is not None else np.zeros(n), dtype=torch.float64)
    bang = torch.tensor(bang if bang is not None else np.zeros(n, bool), dtype=torch.bool)
    p = base.scaled(mass=torch.tensor(mass_s, dtype=torch.float64), stiffness=torch.tensor(stiff_s, dtype=torch.float64))
    g = p.g
    vlo, vhi = base.ux_target_range
    sub = int(round(dt_save / dt_sim))
    n_save = int(round(duration / dt_save))

    ou_steer = OU(n, tau=1.0 if base.name.startswith("full") else 0.5, sigma=1.2, dt=dt_save, gen=gen)
    ou_speed = OU(n, tau=5.0, sigma=1.0, dt=dt_save, gen=gen)
    ou_long = OU(n, tau=0.5, sigma=1.0, dt=dt_save, gen=gen)
    kp = 1.0  # 속도 P 제어 이득 [1/s]

    to_v = lambda s: vlo + (vhi - vlo) * torch.sigmoid(s)
    x = torch.zeros(n, 6, dtype=torch.float64)
    x[:, 3] = to_v(ou_speed.x)
    delta = torch.zeros(n, dtype=torch.float64)

    rec = torch.full((n, n_save, len(CHANNELS)), float("nan"), dtype=torch.float64)
    alive = torch.ones(n, dtype=torch.bool)
    length = torch.full((n,), n_save, dtype=torch.long)
    reason = ["full"] * n
    ap_f, ap_r = peak_slip(p.tire_f, mu), peak_slip(p.tire_r, mu)
    dead_state = torch.tensor([0, 0, 0, 1.0, 0, 0], dtype=torch.float64).expand(n, 6)

    for i in range(n_save):
        ux = x[:, 3]
        # 조향 명령
        s = torch.tanh(ou_steer.step())          # 조향 세기 [-1, 1]
        ay_cmd = k * mu * g * s
        d_cmd = torch.atan(p.L * ay_cmd / torch.clamp(ux, min=1e-3) ** 2)
        d_bang = torch.where(torch.abs(s) > 0.3, p.delta_max * torch.sign(s), torch.zeros_like(s))
        d_cmd = torch.where(bang, d_bang, d_cmd)
        d_cmd = torch.clamp(d_cmd, -p.delta_max, p.delta_max)
        step = p.delta_rate_max * dt_save
        delta = delta + torch.clamp(d_cmd - delta, -step, step)
        # 종방향 명령
        v_t = to_v(ou_speed.step())
        low = torch.clamp(ux / vlo, max=1.0)     # 저속일수록 랜덤 성분을 줄인다
        fx = p.m * (kp * (v_t - ux) + 0.3 * mu * g * low * ou_long.step()
                    + cpl * mu * g * torch.abs(s) * low)
        fx_max = torch.minimum(p.power_max / torch.clamp(ux, min=0.5), mu * p.m * g)
        fx = torch.maximum(torch.minimum(fx, fx_max), -mu * p.m * g)
        fx = torch.where(ux < 2 * p.ux_min_valid, torch.clamp(fx, min=0.0), fx)  # 정지 직전엔 제동 금지
        u = torch.stack([delta, fx], -1)

        xd, f = derivatives(x, u, p, mu, opt, return_forces=True)
        ratio = torch.maximum(torch.abs(f["alpha_f"]) / ap_f, torch.abs(f["alpha_r"]) / ap_r)
        row = torch.stack([torch.full_like(ux, i * dt_save), *x.unbind(-1), delta, fx,
                           f["fxf"], f["fxr"], f["fyf"], f["fyr"], f["fzf"], f["fzr"], f["alpha_f"], f["alpha_r"],
                           xd[:, 3], xd[:, 4], xd[:, 5], f["ax"], f["ay"], ratio], -1)
        rec[alive, i] = row[alive]

        for _ in range(sub):
            x = rk4_step(x, u, p, mu, dt_sim, opt)

        # 종료 조건: 차체 슬립각 30° 초과(스핀), 최저 속도 미만, 수치 발산
        beta = torch.atan2(x[:, 4], x[:, 3])
        bad = ~torch.isfinite(x).all(-1)
        spin = torch.abs(beta) > math.radians(30)
        slow = x[:, 3] < p.ux_min_valid
        ended = alive & (bad | spin | slow)
        for j in torch.nonzero(ended).flatten().tolist():
            reason[j] = "diverged" if bad[j] else "spin" if spin[j] else "slow"
            length[j] = i + 1
        alive &= ~ended
        # 끝난 에피소드는 안전한 더미 상태로 바꿔서 계속 배치 계산이 터지지 않게 한다(기록은 안 됨)
        x = torch.where(alive[:, None], x, dead_state)
        if not alive.any():
            break
    return rec, length, reason


def add_sensor_noise(ep, std, rng):
    """가우시안 잡음 + 에피소드마다 고정된 작은 편향(잡음 표준편차의 20%)."""
    for c in MEAS:
        s = std.get(c, 0.0)
        ep["meas_" + c] = ep[c] + rng.normal(0, 0.2 * s) + rng.normal(0, s, size=ep[c].shape)
    return ep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", choices=list(PRESETS), default="f1tenth")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--dt-sim", type=float, default=0.001)
    ap.add_argument("--dt-save", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--k-range", type=float, nargs=2, default=(0.6, 1.2))
    ap.add_argument("--mass-jitter", type=float, default=0.10)
    ap.add_argument("--stiffness-jitter", type=float, default=0.15)
    ap.add_argument("--frac-interp", type=float, default=0.15)
    ap.add_argument("--frac-extrap", type=float, default=0.05)
    ap.add_argument("--mode-probs", type=float, nargs=3, default=(0.4, 0.4, 0.2), help="random power brake 비율")
    ap.add_argument("--coupling-range", type=float, nargs=2, default=(0.3, 1.0), help="power/brake 결합 세기(μg 단위)")
    ap.add_argument("--bang-frac", type=float, default=None, help="뱅뱅 조향 에피소드 비율 (기본: 소형 0.5, 풀스케일 0)")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    base = PRESETS[args.vehicle]
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    os.makedirs(os.path.join(args.out, "episodes"), exist_ok=True)

    plan = assign_episodes(args.episodes, rng, args.frac_interp, args.frac_extrap)
    k_aggr = rng.uniform(*args.k_range, size=args.episodes)
    mass_s = rng.uniform(1 - args.mass_jitter, 1 + args.mass_jitter, size=args.episodes)
    stiff_s = rng.uniform(1 - args.stiffness_jitter, 1 + args.stiffness_jitter, size=args.episodes)
    probs = np.array(args.mode_probs) / np.sum(args.mode_probs)
    modes = rng.choice(MODES, size=args.episodes, p=probs)
    sign = np.select([modes == "power", modes == "brake"], [1.0, -1.0], 0.0)
    coupling = sign * rng.uniform(*args.coupling_range, size=args.episodes)
    bang_frac = BANG_FRAC_DEFAULT[args.vehicle] if args.bang_frac is None else args.bang_frac
    bang = rng.random(args.episodes) < bang_frac

    rows, t0 = [], time.time()
    for s in range(0, args.episodes, args.batch):
        e = min(s + args.batch, args.episodes)
        rec, length, reason = simulate_batch(base, [plan[i][1] for i in range(s, e)], k_aggr[s:e], mass_s[s:e], stiff_s[s:e],
                                             args.duration, args.dt_sim, args.dt_save, gen,
                                             coupling=coupling[s:e], bang=bang[s:e])
        rec = rec.numpy()
        for j, i in enumerate(range(s, e)):
            n = int(length[j])
            ep = {c: rec[j, :n, ci].astype(np.float32) for ci, c in enumerate(CHANNELS)}
            if not args.no_noise:
                ep = add_sensor_noise(ep, base.noise_std, rng)
                ep = {c: v.astype(np.float32) for c, v in ep.items()}
            fname = f"ep_{i:06d}.npz"
            np.savez_compressed(os.path.join(args.out, "episodes", fname), **ep)
            rows.append(dict(episode=i, file=fname, split=plan[i][0], mu=plan[i][1], k=round(float(k_aggr[i]), 4),
                             mode=str(modes[i]), coupling=round(float(coupling[i]), 4), bang=int(bang[i]),
                             mass_scale=round(float(mass_s[i]), 4), stiffness_scale=round(float(stiff_s[i]), 4),
                             n_samples=n, end_reason=reason[j],
                             frac_saturated=round(float(np.mean(ep["regime_ratio"] > 1.0)), 4)))
        print(f"{e}/{args.episodes} episodes, {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cfg = dict(args=vars(args), vehicle=asdict(base), mu_sets=MU_SETS, channels=CHANNELS, meas=["meas_" + c for c in MEAS],
               regime_bins={"normal": "<0.5", "nonlinear": "0.5-1.0", "saturated": ">1.0"},
               model="P2 (pacejka, load transfer, friction ellipse)")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)
    print(f"done: {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
