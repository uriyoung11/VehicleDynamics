"""물리 구조를 넣은 학습 모델 M1, M2, M3. 입출력은 M0·B1과 같다.

  pred = model(hist)
    hist: (B, T, 5) [r, Uy, Ux, delta, long/long_scale], 마지막 행이 현재 시점
    pred: (B, 2) 다음 시점(dt 뒤) [r, Uy]

물리의 양: M0(없음) < M3(손실에만) < M2(동역학 식) < M1(동역학 + 타이어 식) < B1(전부, 신경망 없음)

공통 물리(PhysCore)
  B1과 같은 P1 자전거 모델(Fiala 타이어, 하중 이동·결합 슬립 없음). 앞뒤 코너링 강성과 μ 배율을 학습한다.
  생성기(P2)보다 단순한 물리를 일부러 쓴다(inverse crime 방지, project.md 3-1).
  종방향 입력이 모터 전류(D2)면 "구동력 = k · 전류"의 k도 같이 학습한다. 그래서 M1·M2는 D2에서도 돈다.

M1 물리 잔차 (Kabzan et al. 2019 RA-L, Spielberg 박사논문, D2 논문의 PCR)
  상태 미분 = P1 물리 미분 + 신경망 잔차.  잔차 신경망의 마지막 층을 0으로 시작해 처음엔 B1과 같다.
  물리 파라미터는 신경망보다 큰 학습률로 먼저 맞춰지게 한다(param_groups).

M2 신경망 타이어 (D2 논문 NNTF, Neural-ExpTanh(Djeumou et al. 2023), Kim et al. 2022)
  운동 방정식은 고정하고 앞뒤 타이어 횡력만 신경망이 낸다. 타이어 기본 성질을 구조로 보장한다.
    Fy = -μ̂ · Fz · tanh( c·α + g(α, ·) - g(-α, ·) )
    - α에 대해 홀함수 → 슬립 0이면 힘 0, 좌우 대칭
    - tanh 상한 → |Fy| <= μ̂·Fz (마찰 한계. ExpTanh 논문은 벌점으로, 여기선 구조로)
    - c·α 항 → 학습 시작점이 "공칭 코너링 강성을 가진 포화 타이어"
    - μ̂: 과거 이력에서 노면 마찰을 추정하는 작은 신경망. 주 실험은 μ를 주지 않으므로(11/7 결정)
      D2 논문 NNTF가 UKF 상태로 μ를 추정한 것을 이력 기반 추정으로 대신한다.
  타이어 신경망 입력: 슬립각, 그 축 종방향 힘/Fz(결합 슬립), 종속도.

M3 물리 손실 (PINN: Raissi et al. 2019, 차량 적용 예 arXiv 2303.00474)
  구조는 M0 그대로. 학습할 때만 물리 벌점을 더한다.
  (a) 제약 벌점: 예측한 미분을 운동 방정식으로 거꾸로 풀어 "타이어가 냈어야 할 횡력"을 구하고,
      |Fy| > μ_max·Fz(마찰 한계)이거나 Fy가 슬립과 같은 방향이면(부호 규약, ExpTanh 논문과 같음) 벌점.
      타이어 모양을 가정하지 않는다.
  (b) 모델 잔차 벌점: 예측 미분과 P1 물리 미분의 차이(가중치 작게). 물리 파라미터는 같이 학습한다.
  벌점은 학습 데이터와, 데이터 상태를 흔든 라벨 없는 점(collocation)에 모두 준다.
  추론 때는 M0와 똑같이 신경망만 쓴다.
"""
import math
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics import PRESETS, ModelOptions, derivatives
from models.baselines import M0MLP

P1_FIALA = ModelOptions(tire="fiala", load_transfer=False, combined=False, resistance=False)
UX_MIN = 0.5   # 슬립각 계산에서 0으로 나누기 방지 [m/s]


def mlp(n_in, hidden, n_out, act=nn.Softplus, zero_last=False):
    layers, n = [], n_in
    for h in hidden:
        layers += [nn.Linear(n, h), act()]; n = h
    layers.append(nn.Linear(n, n_out))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    if zero_last:
        nn.init.zeros_(net[-1].weight)
    return net


def _buf(v, default):
    return torch.as_tensor(v if v is not None else default, dtype=torch.float64)


class PhysCore(nn.Module):
    """B1과 같은 P1 물리 + 종방향 입력 해석. 학습 파라미터: 코너링 강성·μ 배율(로그), 모터 이득(로그)."""

    def __init__(self, vehicle, long_kind="force_front", long_scale=1.0, motor_k0=1.0):
        super().__init__()
        self.vehicle, self.long_kind = vehicle, long_kind
        self.register_buffer("long_scale", torch.tensor(float(long_scale), dtype=torch.float64))
        self.log_cf = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_cr = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_mu = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_k = nn.Parameter(torch.tensor(math.log(max(motor_k0, 1e-6)), dtype=torch.float64),
                                  requires_grad=(long_kind == "motor_current"))

    @property
    def base(self):
        return PRESETS[self.vehicle]

    def front_share(self):
        # 입력이 앞바퀴 힘이면 전부 앞, 모터 전류면 차량의 구동 배분
        return 1.0 if self.long_kind == "force_front" else float(self.base.drive_front)

    def total_fx(self, long):
        x = long * self.long_scale
        return x if self.long_kind == "force_front" else torch.exp(self.log_k) * x

    def params(self):
        b, s = self.base, self.front_share()
        return replace(b, drive_front=s, brake_front=s,
                       tire_f=replace(b.tire_f, c_alpha=b.tire_f.c_alpha * torch.exp(self.log_cf)),
                       tire_r=replace(b.tire_r, c_alpha=b.tire_r.c_alpha * torch.exp(self.log_cr)))

    def mu(self):
        return torch.exp(self.log_mu)

    def deriv(self, last):
        """last: (B, 5) 현재 시점. 반환 (B, 2) [r_dot, Uy_dot]."""
        r, uy, ux, delta = last[:, 0], last[:, 1], torch.clamp(last[:, 2], min=UX_MIN), last[:, 3]
        z = torch.zeros_like(r)
        xd = derivatives(torch.stack([z, z, z, ux, uy, r], -1), torch.stack([delta, self.total_fx(last[:, 4])], -1),
                         self.params(), self.mu(), P1_FIALA)
        return torch.stack([xd[:, 5], xd[:, 4]], -1)

    def fitted(self):
        with torch.no_grad():
            p = self.params()
            out = dict(c_alpha_f=p.tire_f.c_alpha.item(), c_alpha_r=p.tire_r.c_alpha.item(), mu=self.mu().item())
            if self.long_kind == "motor_current":
                out["motor_k"] = torch.exp(self.log_k).item()
            return out


class _Standardize(nn.Module):
    def __init__(self, x_mean, x_std):
        super().__init__()
        self.register_buffer("x_mean", _buf(x_mean, [0.0])); self.register_buffer("x_std", _buf(x_std, [1.0]))

    def forward(self, hist):
        return (hist.reshape(len(hist), -1) - self.x_mean) / self.x_std


# ---------------------------------------------------------------- M1

class M1Residual(nn.Module):
    def __init__(self, vehicle, long_kind="force_front", long_scale=1.0, motor_k0=1.0, T=4, dt=0.01,
                 hidden=(128, 128), x_mean=None, x_std=None, d_mean=None, d_std=None):
        super().__init__()
        self.T, self.dt = T, dt
        self.phys = PhysCore(vehicle, long_kind, long_scale, motor_k0)
        self.std = _Standardize(x_mean, x_std)
        self.net = mlp(5 * T, hidden, 2, zero_last=True)
        self.register_buffer("d_std", _buf(d_std, [1.0]))

    def forward(self, hist):
        last = hist[:, -1]
        d = self.phys.deriv(last) + self.net(self.std(hist)) * self.d_std
        return last[:, :2] + self.dt * d

    def param_groups(self, lr):
        return [dict(params=self.net.parameters(), lr=lr), dict(params=self.phys.parameters(), lr=10 * lr)]

    def fitted(self):
        return self.phys.fitted()


# ---------------------------------------------------------------- M2

class _AxleTire(nn.Module):
    """한 축의 타이어 횡력 계수 g ∈ (-1, 1). Fy = -μ̂·Fz·g. α에 대해 홀함수, 상한 1."""

    def __init__(self, c_alpha, hidden=(64, 64)):
        super().__init__()
        self.c = c_alpha                       # 하중 정규화 코너링 강성 [1/rad] (μ로 나눈 값은 forward에서)
        self.net = mlp(3, hidden, 1, act=nn.Tanh, zero_last=True)

    def forward(self, alpha, fx_n, ux_n, mu):
        g = lambda a: self.net(torch.stack([a / 0.1, fx_n, ux_n], -1))[:, 0]
        return torch.tanh(self.c / mu * alpha + g(alpha) - g(-alpha))


class M2TireNN(nn.Module):
    def __init__(self, vehicle, long_kind="force_front", long_scale=1.0, motor_k0=1.0, T=4, dt=0.01,
                 hidden=(80, 80), mu_hidden=(64, 64), x_mean=None, x_std=None, ux_scale=None, **_):
        super().__init__()
        self.T, self.dt = T, dt
        self.vehicle = vehicle
        self.phys = PhysCore(vehicle, long_kind, long_scale, motor_k0)   # 종방향 입력 해석(모터 이득)만 사용
        self.std = _Standardize(x_mean, x_std)
        p = PRESETS[vehicle]
        self.tire_f, self.tire_r = _AxleTire(p.tire_f.c_alpha, hidden), _AxleTire(p.tire_r.c_alpha, hidden)
        self.mu_net = mlp(5 * T, mu_hidden, 1, zero_last=True)
        self.register_buffer("ux_scale", torch.tensor(float(ux_scale or p.ux_target_range[1]), dtype=torch.float64))
        self.mu_bias = math.log(math.e - 1)    # softplus(mu_bias) = 1 → 처음 μ̂ = 1

    def mu_hat(self, hist):
        return 0.05 + F.softplus(self.mu_net(self.std(hist))[:, 0] + self.mu_bias)

    def forces(self, hist):
        p = PRESETS[self.vehicle]
        last = hist[:, -1]
        r, uy, ux, delta = last[:, 0], last[:, 1], torch.clamp(last[:, 2], min=UX_MIN), last[:, 3]
        fx = self.phys.total_fx(last[:, 4])
        s = self.phys.front_share()
        fxf, fxr = s * fx, (1 - s) * fx
        fzf, fzr = p.m * p.g * p.b / p.L, p.m * p.g * p.a / p.L
        af = torch.atan((uy + p.a * r) / ux) - delta
        ar = torch.atan((uy - p.b * r) / ux)
        mu = self.mu_hat(hist)
        ux_n = ux / self.ux_scale
        fyf = -mu * fzf * self.tire_f(af, fxf / fzf, ux_n, mu)
        fyr = -mu * fzr * self.tire_r(ar, fxr / fzr, ux_n, mu)
        return dict(fyf=fyf, fyr=fyr, fxf=fxf, af=af, ar=ar, mu=mu)

    def forward(self, hist):
        p = PRESETS[self.vehicle]
        last = hist[:, -1]
        r, ux, delta = last[:, 0], torch.clamp(last[:, 2], min=UX_MIN), last[:, 3]
        f = self.forces(hist)
        lat_f = f["fyf"] * torch.cos(delta) + f["fxf"] * torch.sin(delta)
        uy_dot = (lat_f + f["fyr"]) / p.m - r * ux
        r_dot = (p.a * lat_f - p.b * f["fyr"]) / p.Iz
        return last[:, :2] + self.dt * torch.stack([r_dot, uy_dot], -1)

    def param_groups(self, lr):
        return [dict(params=[*self.tire_f.parameters(), *self.tire_r.parameters(), *self.mu_net.parameters()], lr=lr),
                dict(params=self.phys.parameters(), lr=10 * lr)]

    def fitted(self):
        out = {k: v for k, v in self.phys.fitted().items() if k == "motor_k"}
        return out or None


# ---------------------------------------------------------------- M3

class M3PINN(M0MLP):
    def __init__(self, vehicle, long_kind="force_front", long_scale=1.0, motor_k0=1.0, T=4, dt=0.01, hidden=(128, 128),
                 x_mean=None, x_std=None, d_mean=None, d_std=None, mu_max=1.2, lam_c=1.0, lam_m=0.1,
                 colloc=1.0, colloc_sigma=0.5, alpha_dead=0.02, state_std=None):
        super().__init__(T=T, hidden=hidden, dt=dt, x_mean=x_mean, x_std=x_std, d_mean=d_mean, d_std=d_std)
        self.vehicle = vehicle
        self.phys = PhysCore(vehicle, long_kind, long_scale, motor_k0)
        self.mu_max, self.lam_c, self.lam_m = mu_max, lam_c, lam_m
        self.colloc, self.colloc_sigma, self.alpha_dead = colloc, colloc_sigma, alpha_dead
        # collocation 흔들기 크기: [r, Uy, delta]의 데이터 표준편차
        self.register_buffer("state_std", _buf(state_std, [1.0, 1.0, 1.0]))

    def _implied_violation(self, hist, deriv):
        """예측 미분 → 운동 방정식 역산 → 앞뒤 횡력. 마찰 한계·부호 위반을 Fz로 정규화해 돌려준다."""
        p = PRESETS[self.vehicle]
        last = hist[:, -1]
        r, uy, ux, delta = last[:, 0], last[:, 1], torch.clamp(last[:, 2], min=UX_MIN), last[:, 3]
        r_dot, uy_dot = deriv[:, 0], deriv[:, 1]
        fxf = self.phys.front_share() * self.phys.total_fx(last[:, 4])
        S = p.m * (uy_dot + r * ux)                 # 횡방향 합력
        Mz = p.Iz * r_dot                           # 요 모멘트
        lat_f = (p.b * S + Mz) / p.L                # Fyf cosδ + Fxf sinδ
        fyr = (p.a * S - Mz) / p.L
        fyf = (lat_f - fxf * torch.sin(delta)) / torch.cos(delta)
        fzf, fzr = p.m * p.g * p.b / p.L, p.m * p.g * p.a / p.L
        af = torch.atan((uy + p.a * r) / ux) - delta
        ar = torch.atan((uy - p.b * r) / ux)
        v = 0.0
        for fy, fz, a in ((fyf, fzf, af), (fyr, fzr, ar)):
            v = v + F.relu(torch.abs(fy) / fz - self.mu_max) ** 2                       # 마찰 한계
            v = v + (F.relu(fy * torch.sign(a)) / fz * (torch.abs(a) > self.alpha_dead)) ** 2   # 슬립과 같은 방향
        return v.mean()

    def physics_loss(self, X):
        """학습 루프가 데이터 손실에 더하는 벌점. X: 학습 배치 입력."""
        pts = [X]
        if self.colloc > 0:
            n = max(1, int(self.colloc * len(X)))
            Xc = X[torch.randint(len(X), (n,))].clone()
            eps = torch.randn(n, 1, 3, dtype=X.dtype) * self.colloc_sigma * self.state_std
            Xc[:, :, [0, 1, 3]] += eps                 # 이력 전체에 같은 오프셋: r, Uy, delta
            pts.append(Xc)
        Xa = torch.cat(pts)
        deriv = (self(Xa) - Xa[:, -1, :2]) / self.dt
        loss = self.lam_c * self._implied_violation(Xa, deriv)
        if self.lam_m > 0:
            resid = (deriv - self.phys.deriv(Xa[:, -1])) / self.d_std
            loss = loss + self.lam_m * (resid ** 2).mean()
        return loss

    def param_groups(self, lr):
        return [dict(params=self.net.parameters(), lr=lr), dict(params=self.phys.parameters(), lr=10 * lr)]

    def fitted(self):
        return self.phys.fitted()
