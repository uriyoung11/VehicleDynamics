"""MPPI(샘플링 기반 MPC). Williams et al. 2017, "Information theoretic MPC for model-based reinforcement learning".

매 제어 주기(dt_mpc)마다:
  1. 지난 계획을 한 칸 밀어 시작점으로 쓴다(warm start).
  2. 조향·종방향 힘 계획에 잡음을 더한 후보 K개를 만든다.
  3. 차량 모델(dynamics)로 각 후보를 H스텝(= H·dt_mpc 초) 앞까지 굴린다.
  4. 비용 J를 계산하고 w = exp(-(J - min J)/λ)로 가중 평균해 계획을 갱신한다.
  5. 계획의 첫 동작을 실행한다.

모델은 미분이 필요 없으므로 PyTorch 모델(M0~M3, B1)을 고치지 않고 넣을 수 있다.
비용: 중심선 횡오차, 목표 속도 오차, 트랙 가장자리 접근·이탈, 조작 변화량.
"""
import math
from dataclasses import dataclass
import numpy as np
import torch

G = 9.81


@dataclass
class MPPIConfig:
    K: int = 256              # 후보 수
    H: int = 20               # 예측 스텝 수
    dt: float = 0.05          # 제어 주기·예측 간격 [s]
    sigma_delta: float = 0.06  # 조향 잡음 [rad]
    sigma_fx: float = 0.25    # 종방향 힘 잡음 [m·g 배]
    lam: float = 1.0          # 온도. 작을수록 가장 좋은 후보에 몰린다
    w_lat: float = 4.0        # 중심선 횡오차 [1/m²]
    w_v: float = 1.0          # 속도 오차 [1/(m/s)²]
    w_edge: float = 400.0     # 가장자리 접근 (여유 < edge_margin)
    edge_margin: float = 0.25  # [m]
    w_off: float = 2000.0     # 트랙 이탈(여유 < 0), 이탈 거리에 비례
    w_ddelta: float = 20.0    # 조향 변화 [1/rad²]
    w_dfx: float = 0.5        # 종방향 힘 변화 [1/(m·g)²]
    # 방식 B (진행 최대화) 전용
    w_prog: float = 20.0      # 예측 구간 동안 중심선을 따라 전진한 거리 보상 [1/m]
    w_lat_b: float = 0.5      # 방식 B의 약한 중심선 횡오차 [1/m²]
    v_max: float = 7.0        # 속도 상한 [m/s]. 학습 데이터 속도 범위(~6 m/s)를 크게 벗어나지 않게, 방식 A의 v_max와 같게 둔다
    w_vmax: float = 50.0      # 속도 상한 초과 벌점 [1/(m/s)²]
    seed: int = 0


class TrackCost:
    """트랙 중심선 기준 비용. 예측 점마다 현재 위치 근처 창에서 가장 가까운 중심선 점을 찾는다.

    mode="track"    방식 A. 목표 속도 프로파일 추종: 횡오차² + (속도 − 목표 속도)² + 가장자리·이탈 벌점
    mode="progress" 방식 B. 진행 최대화: −(예측 구간 동안 전진 거리) + 가장자리·이탈 벌점 + 약한 횡오차
                    목표 속도가 없다. 얼마나 빨리 갈지는 예측 모델이 "트랙을 벗어나지 않는다"고 보는 범위에서 정해진다.
    """

    def __init__(self, tr, v_ref, cfg, window=(-5, 60), mode="track"):
        assert mode in ("track", "progress")
        c = tr["center"]
        t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
        T = lambda a: torch.as_tensor(a, dtype=torch.float64)
        self.c, self.n = T(c), T(np.stack([-t[:, 1], t[:, 0]], 1))
        self.wl, self.wr = T(np.linalg.norm(tr["left"] - c, axis=1)), T(np.linalg.norm(tr["right"] - c, axis=1))
        self.v_ref = T(v_ref)
        seg = np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1)
        self.s, self.L = T(np.r_[0, np.cumsum(seg)[:-1]]), float(seg.sum())   # 중심선 점의 누적 거리
        self.N, self.cfg, self.win, self.mode = len(c), cfg, np.arange(*window), mode

    def nearest(self, xy, idx0):
        """xy: (..., 2) → 가장 가까운 중심선 점 번호 (...)."""
        w = torch.as_tensor((idx0 + self.win) % self.N)
        d = ((xy[..., None, :] - self.c[w]) ** 2).sum(-1)
        return w[d.argmin(-1)]

    def margin(self, xy, j):
        e = ((xy - self.c[j]) * self.n[j]).sum(-1)      # 중심선 왼쪽(+) 오프셋
        return e, torch.minimum(self.wl[j] - e, self.wr[j] + e)

    def stage(self, xy, ux, idx0):
        cfg = self.cfg
        j = self.nearest(xy, idx0)
        e, m = self.margin(xy, j)
        if self.mode == "track":
            c = cfg.w_lat * e ** 2 + cfg.w_v * (ux - self.v_ref[j]) ** 2
        else:
            c = cfg.w_lat_b * e ** 2 + cfg.w_vmax * torch.relu(ux - cfg.v_max) ** 2
        c = c + cfg.w_edge * torch.relu(cfg.edge_margin - m) ** 2 + cfg.w_off * torch.relu(-m)
        return c

    def terminal(self, xy, idx0):
        """예측 끝 위치의 비용. 방식 B: 중심선을 따라 전진한 거리만큼 보상(음수 비용)."""
        if self.mode == "track":
            return torch.zeros(xy.shape[0], dtype=torch.float64)
        j = self.nearest(xy, idx0)
        ds = torch.remainder(self.s[j] - self.s[idx0 % self.N] + self.L / 2, self.L) - self.L / 2
        return -self.cfg.w_prog * ds


class MPPI:
    def __init__(self, dynamics, cost, p, cfg=MPPIConfig()):
        """dynamics: rollout(state, U) -> (K, H, 4) [X, Y, psi, ux] 를 가진 객체 (아래 모델 래퍼).
        p: 차량 파라미터(조향 한계, 질량 등)."""
        self.dyn, self.cost, self.p, self.cfg = dynamics, cost, p, cfg
        self.U = torch.zeros(cfg.H, 2, dtype=torch.float64)   # [조향 명령, 종방향 힘/(m·g)]
        self.gen = torch.Generator().manual_seed(cfg.seed)

    def reset(self, delta0=0.0):
        self.U[:] = 0.0; self.U[:, 0] = delta0

    def step(self, state, idx0, mu_lim):
        """state: 모델 래퍼가 이해하는 현재 상태(dict). idx0: 현재 가장 가까운 중심선 점. mu_lim: 힘 제한용 μ."""
        cfg, p = self.cfg, self.p
        self.U = torch.cat([self.U[1:], self.U[-1:]])          # warm start
        eps = torch.randn(cfg.K, cfg.H, 2, generator=self.gen, dtype=torch.float64)
        eps[..., 0] *= cfg.sigma_delta; eps[..., 1] *= cfg.sigma_fx
        eps[0] = 0.0                                           # 후보 하나는 현재 계획 그대로
        V = self.U[None] + eps
        V[..., 0] = V[..., 0].clamp(-p.delta_max, p.delta_max)
        V[..., 1] = V[..., 1].clamp(-mu_lim, mu_lim)
        traj, d_applied = self.dyn.rollout(state, V)           # (K, H, 4), (K, H)
        J = torch.zeros(cfg.K, dtype=torch.float64)
        for k in range(cfg.H):
            J += self.cost.stage(traj[:, k, :2], traj[:, k, 3], idx0)
        J += self.cost.terminal(traj[:, -1, :2], idx0)
        dd = torch.diff(torch.cat([torch.full((cfg.K, 1), float(state["delta"]), dtype=torch.float64), d_applied], 1), dim=1)
        J += cfg.w_ddelta * (dd ** 2).sum(1) + cfg.w_dfx * (torch.diff(V[..., 1], dim=1) ** 2).sum(1)
        J = torch.nan_to_num(J, nan=1e9, posinf=1e9)
        w = torch.exp(-(J - J.min()) / cfg.lam); w /= w.sum()
        self.U = (w[:, None, None] * V).sum(0)
        best = int(J.argmin())
        return self.U[0].clone(), dict(J_min=float(J.min()), ess=float(1.0 / (w ** 2).sum()), plan=traj[best, :, :2].numpy())


# ---------------------------------------------------------------- 모델 래퍼

def _clip_fx(fx_n, ux, p, mu):
    """종방향 힘 명령 [m·g 배] → [N]. 출력 한계와 마찰 한계."""
    fx = fx_n * p.m * G
    return torch.minimum(fx, torch.as_tensor(p.power_max, dtype=torch.float64) / torch.clamp(ux, min=0.5)).clamp(-mu * p.m * G, mu * p.m * G)


def _steer(delta, cmd, p, dt):
    return delta + torch.clamp(cmd - delta, -p.delta_rate_max * dt, p.delta_rate_max * dt)


class PhysicsDynamics:
    """물리 모델(P0~P2)로 굴린다. mu_fn(xy) -> 그 위치에서 모델이 믿는 μ."""

    def __init__(self, p, opt, mu_fn, dt_int=0.01):
        from physics import rk4_step
        self.p, self.opt, self.mu_fn, self.dt_int, self.rk4 = p, opt, mu_fn, dt_int, rk4_step

    def rollout(self, state, V):
        K, H, _ = V.shape
        dt = 0.05 if not hasattr(self, "dt_mpc") else self.dt_mpc
        x = torch.as_tensor(state["x"], dtype=torch.float64).repeat(K, 1)       # (K, 6)
        delta = torch.full((K,), float(state["delta"]), dtype=torch.float64)
        n_sub = max(1, int(round(dt / self.dt_int)))
        out, dl = torch.zeros(K, H, 4, dtype=torch.float64), torch.zeros(K, H, dtype=torch.float64)
        for k in range(H):
            mu = self.mu_fn(x[:, :2])
            fx = _clip_fx(V[:, k, 1], x[:, 3], self.p, mu)
            for _ in range(n_sub):
                delta = _steer(delta, V[:, k, 0], self.p, self.dt_int)
                x = self.rk4(x, torch.stack([delta, fx], -1), self.p, mu, self.dt_int, self.opt)
            out[:, k] = x[:, :4]; dl[:, k] = delta
        return out, dl


class LearnedDynamics:
    """학습 모델(hist -> 다음 [r, Uy])로 굴린다. 모델 시간 간격(0.01 s)으로 여러 번 호출한다.
    Ux는 모든 모델에 같은 물리식을 쓴다: dUx/dt = (Fx - 주행 저항)/m + r·Uy  (docs/mpc_plan 6절)."""

    def __init__(self, model, p, long_scale, front_share, mu_fn=None):
        """mu_fn: 주면 종방향 힘 한계(제동·가속)를 위치별 μ로 자른다(진단용: 정답 μ 지도). 없으면 state["mu_lim"] 고정값."""
        self.model, self.p, self.long_scale, self.front_share, self.mu_fn = model, p, long_scale, front_share, mu_fn

    @torch.no_grad()
    def rollout(self, state, V):
        p = self.p
        K, H, _ = V.shape
        dt_m = self.model.dt
        n_sub = int(round(0.05 / dt_m))
        hist = torch.as_tensor(state["hist"], dtype=torch.float64).repeat(K, 1, 1)   # (K, T, 5)
        X, Y, psi = (torch.full((K,), float(state["x"][i]), dtype=torch.float64) for i in range(3))
        delta = hist[:, -1, 3].clone()
        out, dl = torch.zeros(K, H, 4, dtype=torch.float64), torch.zeros(K, H, dtype=torch.float64)
        for k in range(H):
            for _ in range(n_sub):
                r, uy, ux = hist[:, -1, 0], hist[:, -1, 1], hist[:, -1, 2]
                mu_l = self.mu_fn(torch.stack([X, Y], -1)) if self.mu_fn is not None else state["mu_lim"]
                fx = _clip_fx(V[:, k, 1], ux, p, mu_l)
                delta = _steer(delta, V[:, k, 0], p, dt_m)
                fxf = torch.where(fx > 0, self.front_share[0] * fx, self.front_share[1] * fx)
                # 현재 입력을 이력 마지막 행에 넣고 다음 [r, Uy] 예측
                cur = torch.stack([r, uy, ux, delta, fxf / self.long_scale], -1)
                h = torch.cat([hist[:, :-1], cur[:, None]], 1)
                nxt = self.model(h)
                f_res = p.c0 + p.c1 * ux + p.c2 * ux ** 2
                ux_n = ux + dt_m * ((fx - f_res) / p.m + r * uy)
                X = X + dt_m * (ux * torch.cos(psi) - uy * torch.sin(psi))
                Y = Y + dt_m * (ux * torch.sin(psi) + uy * torch.cos(psi))
                psi = psi + dt_m * r
                new = torch.stack([nxt[:, 0], nxt[:, 1], ux_n, delta, fxf / self.long_scale], -1)
                hist = torch.cat([h[:, 1:], new[:, None]], 1)
            out[:, k] = torch.stack([X, Y, psi, hist[:, -1, 2]], -1); dl[:, k] = delta
        return out, dl
