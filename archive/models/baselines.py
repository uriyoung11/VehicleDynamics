"""기준 모델 M0, B1. 모든 모델은 같은 인터페이스를 쓴다.

  pred = model(hist)
    hist: (B, T, 5) [r, Uy, Ux, delta, long/long_scale], 시간 순서. 마지막 행이 현재 시점.
    pred: (B, 2) 다음 시점(dt 뒤) [r, Uy]

M0: Spielberg 2019 MLP (128x2 softplus, 출력 = 현재 [r, Uy] + dt * 신경망).
    원논문과 다른 점: 입력 표준화와 출력 스케일을 넣었다(D1, D2, D3의 단위가 달라서).
B1: 피팅한 물리 모델. physics의 P1 자전거 모델(Fiala 또는 Pacejka)에서 앞뒤 코너링 강성과 μ만
    데이터로 맞춘다(원논문 Bike_Model과 같은 방식). 종방향 입력이 앞바퀴 힘인 데이터(D1, D3)만 가능.
"""
from dataclasses import replace
import torch
import torch.nn as nn

from physics import PRESETS, ModelOptions, derivatives


class M0MLP(nn.Module):
    def __init__(self, T=4, hidden=(128, 128), dt=0.01, x_mean=None, x_std=None, d_mean=None, d_std=None, n_out=2):
        super().__init__()
        self.T, self.dt, self.n_out = T, dt, n_out   # n_out 2: [r, Uy], 3: [r, Uy, Ux]
        # 입력 채널 수는 표준화 통계 길이에서 정한다(기본 5채널, 차 구분 채널을 붙이면 6)
        layers, n = [], (len(x_mean) if x_mean is not None else 5 * T)
        for h in hidden:
            layers += [nn.Linear(n, h), nn.Softplus()]; n = h
        layers.append(nn.Linear(n, n_out))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        z = lambda v, k: torch.as_tensor(v if v is not None else ([0.0] if "mean" in k else [1.0]), dtype=torch.float64)
        self.register_buffer("x_mean", z(x_mean, "mean")); self.register_buffer("x_std", z(x_std, "std"))
        self.register_buffer("d_mean", z(d_mean, "mean")); self.register_buffer("d_std", z(d_std, "std"))

    def forward(self, hist):
        x = (hist.reshape(len(hist), -1) - self.x_mean) / self.x_std
        deriv = self.net(x) * self.d_std + self.d_mean
        return hist[:, -1, :self.n_out] + self.dt * deriv


class B1Bike(nn.Module):
    def __init__(self, vehicle="fullscale", tire="fiala", long_scale=1.0, dt=0.01, T=4, n_out=2):
        super().__init__()
        self.vehicle, self.tire, self.dt, self.T, self.n_out = vehicle, tire, dt, T, n_out
        self.register_buffer("long_scale", torch.tensor(float(long_scale), dtype=torch.float64))
        # 배율을 로그로 학습(항상 양수)
        self.log_cf = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_cr = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_mu = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.opt = ModelOptions(tire=tire, load_transfer=False, combined=False, resistance=False)

    def params(self):
        base = PRESETS[self.vehicle]
        # 입력이 앞바퀴 힘이므로 구동·제동 모두 앞바퀴 100%로 둔다
        return replace(base, drive_front=1.0, brake_front=1.0,
                       tire_f=replace(base.tire_f, c_alpha=base.tire_f.c_alpha * torch.exp(self.log_cf)),
                       tire_r=replace(base.tire_r, c_alpha=base.tire_r.c_alpha * torch.exp(self.log_cr)))

    def mu(self):
        return torch.exp(self.log_mu)  # 공칭 μ=1 기준 배율. 실제 노면 μ를 곱해서 쓰는 건 2단계에서

    def fitted(self):
        with torch.no_grad():
            p = self.params()
            return dict(c_alpha_f=p.tire_f.c_alpha.item(), c_alpha_r=p.tire_r.c_alpha.item(), mu=self.mu().item())

    def forward(self, hist):
        last = hist[:, -1]
        r, uy, ux, delta, fx = last[:, 0], last[:, 1], last[:, 2], last[:, 3], last[:, 4] * self.long_scale
        z = torch.zeros_like(r)
        x = torch.stack([z, z, z, ux, uy, r], -1)
        u = torch.stack([delta, fx], -1)
        xd = derivatives(x, u, self.params(), self.mu(), self.opt)
        out = [r + self.dt * xd[:, 5], uy + self.dt * xd[:, 4]]
        if self.n_out == 3:
            out.append(ux + self.dt * xd[:, 3])
        return torch.stack(out, -1)


def build(kind, **kw):
    from models.hybrid import M1Residual, M2TireNN, M3PINN
    return {"m0": M0MLP, "b1": B1Bike, "m1": M1Residual, "m2": M2TireNN, "m3": M3PINN}[kind](**kw)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build(ck["kind"], **ck["init"]).double()
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck
