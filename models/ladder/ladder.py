"""물리 비중 사다리 R1~R6 차량 모델 (팀 공유용).

[우리 저장소 사본] 원본: 팀원 김석우 ladder_models.zip (2026-11-17). 바꾼 것은 하나뿐:
  VehicleConfig.use_grade=True 이면 입력에 노면 경사(오르막 +, rad) 채널을 하나 더 받아
  dvx 에 −g·sin(경사)를 더한다. 이때 hist 는 (B, H, 7) = [vx, vy, r, lon, delta, bank, grade].
  use_grade=False(기본)이면 원본과 출력이 같다 (models/ladder/tests/check_same_as_team.py).

이 파일에는 '모델'만 있음. 데이터 읽기·트랙·시뮬레이터·제어기(MPPI)와는 무관함.
차량이나 데이터가 바뀌면 VehicleConfig 하나만 바꿔 끼우면 됨 (configs.py 참고).

상태 s = [vx, vy, r]            (m/s, m/s, rad/s)  vx 앞(+), vy 왼쪽(+), r 반시계(+)
입력 u = [lon, delta, bank]     lon  : 종방향 입력. cfg.lon_input 에 따라 뜻이 다름
                                        "pedal" → 스로틀/브레이크 비율 (-1~1, 브레이크는 음수)
                                        "force" → 앞바퀴 종방향 힘 Fxf (N, 제동은 음수)
                                delta: 앞바퀴 조향각 (rad)
                                bank : 노면 횡경사 (rad). 없으면 0 으로 채움
모델 입력 hist (B, H, 6) = 과거 H 시점의 [vx, vy, r, lon, delta, bank], 오래된 것 → 최근 것.
  마지막 행의 입력이 이번 스텝(cfg.dt 동안)에 가하는 입력.
모델 출력 (B, 3) = cfg.dt 뒤의 [vx, vy, r].

사다리 (아래로 갈수록 물리가 적음)
  R1 순수 물리      : 자전거 모델 + Pacejka 타이어 + 구동·저항 식, 계수는 데이터 전체에 하나(전역)
  R2 물리+잔차      : R1 결과에 MLP(이력) 보정량을 더함
  R3 신경망 타이어  : 자전거 식 그대로, 앞·뒤 타이어 횡력만 MLP (미끄럼각에 대해 홀함수)
  R4 DD식 계수추정  : MLP(이력) → 범위 안 계수(Physics Guard) → R1 식. 계수가 샘플마다 바뀜
  R5 Stanford식 NN  : MLP(이력) → 상태 변화량, 다음 = 현재 + 변화량
  R6 블랙박스       : MLP(이력) → 다음 상태 자체

원본: src/iac_lib.py (AV-21), src/rv_lib.py (Revs Corvette). 두 파일의 모델 부분을 설정만 빼서 합친 것.
pedal 모드에서 원본과 같은 출력을 내는지 tests/check_same_as_repo.py 로 확인함.
"force" 모드(썬더힐 GTI용)는 이 파일에서 새로 넣은 부분이고, 식은 Stanford 데이터 생성 코드
(Model_Learning/data_preprocessing/data_generation_functions.py)의 힘 배분을 따름.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

G = 9.81
NF = 6  # 시점당 특징 수 [vx, vy, r, lon, delta, bank]


@dataclass
class VehicleConfig:
    name: str
    m: float                    # 질량 (kg)
    lf: float                   # 무게중심 → 앞축 (m)
    lr: float                   # 무게중심 → 뒤축 (m)
    phys: list                  # [(이름, 초기값, 하한, 상한), ...]  물리 계수와 Physics Guard 범위
    dt: float = 0.04            # 한 스텝 시간 (s)
    hist: int = 8               # 모델이 보는 과거 시점 수 H
    lon_input: str = "pedal"    # "pedal" 또는 "force"
    brake_front_share: float = 0.66  # force 모드: 제동력 중 앞바퀴 몫 (Stanford b_bias)
    axle_fmax0: tuple = (6000.0, 7000.0)  # R3 앞·뒤 타이어 최대 힘 초기값 (N)
    nsub: int = 4               # 한 스텝을 몇 번에 나눠 적분할지
    note: str = ""
    use_grade: bool = False     # [우리 추가] 노면 경사 채널 사용 여부

    @property
    def nf(self):
        """시점당 특징 수: 6 [vx, vy, r, lon, delta, bank], 경사를 쓰면 7 (+grade)."""
        return NF + 1 if self.use_grade else NF

    @property
    def pnames(self):
        return [p[0] for p in self.phys]

    def check(self):
        need = {"Bf", "Cf", "Df", "Ef", "Br", "Cr", "Dr", "Er", "Cr0", "Cr2", "Iz", "kb", "ds"}
        if self.lon_input == "pedal":
            need |= {"Cm1", "Cm2", "Cb"}
        elif self.lon_input != "force":
            raise ValueError("lon_input 은 'pedal' 또는 'force'")
        miss = need - set(self.pnames)
        if miss:
            raise ValueError(f"{self.name}: phys 에 없는 계수 {sorted(miss)}")
        for n, v0, lo, hi in self.phys:
            if not lo < hi or not lo <= v0 <= hi:
                raise ValueError(f"{self.name}: {n} 범위가 이상함 ({v0}, {lo}, {hi})")


# ---------------------------------------------------------------- 표준화
class Norm(nn.Module):
    """학습 데이터로만 맞춘 표준화 상수 (버퍼로 모델 안에 저장됨).
    X: (N, H, 6) 학습 창, Y: (N, k, 3) 정답 다음 상태. 시험 데이터로 만들면 누수."""

    def __init__(self, X=None, Y=None, nf=NF):
        super().__init__()
        if X is None:
            mu, sd, dsd = torch.zeros(nf), torch.ones(nf), torch.ones(3)
        else:
            X = torch.as_tensor(X, dtype=torch.float64)
            Y = torch.as_tensor(Y, dtype=torch.float64)
            flat = X.reshape(-1, X.shape[-1])
            mu, sd = flat.mean(0), flat.std(0, unbiased=False) + 1e-6
            dsd = (Y[:, 0, :] - X[:, -1, :3]).std(0, unbiased=False) + 1e-9
        self.register_buffer("mu", mu.float())
        self.register_buffer("sd", sd.float())
        self.register_buffer("dsd", dsd.float())  # 1스텝 변화량 표준편차 (손실 가중·출력 크기)

    def x(self, hist):
        return ((hist - self.mu) / self.sd).reshape(len(hist), -1)


# ---------------------------------------------------------------- 물리 부품
def pacejka(alpha, B, C, D, E):
    ba = B * alpha
    return D * torch.sin(C * torch.atan(ba - E * (ba - torch.atan(ba))))


def slips(cfg, vx, vy, r, delta):
    avx = vx.abs().clamp(min=1.0)
    af = delta - torch.atan2(cfg.lf * r + vy, avx)
    ar = torch.atan2(cfg.lr * r - vy, avx)
    return af, ar


def deriv(cfg, vx, vy, r, lon, delta, bank, Ffy, Fry, P, grade=None):
    """자전거 모델 미분. Ffy, Fry: 앞·뒤 타이어 횡력 (N). grade: 노면 경사 rad (우리 추가, None 이면 평지)."""
    m, lf, lr = cfg.m, cfg.lf, cfg.lr
    if cfg.lon_input == "pedal":
        # 원본(iac_lib.frx, bike_deriv)과 같은 식. 구동·제동·저항을 합친 Fx 가 차체 진행 방향으로 작용
        tp, tn = lon.clamp(min=0), lon.clamp(max=0)
        Fx = (P["Cm1"] - P["Cm2"] * vx) * tp + P["Cb"] * tn - P["Cr0"] - P["Cr2"] * vx * vx
        dvx = (Fx - Ffy * torch.sin(delta)) / m + vy * r
        dvy = (Fry + Ffy * torch.cos(delta)) / m - vx * r + P["kb"] * G * torch.sin(bank)
        dr = (Ffy * lf * torch.cos(delta) - Fry * lr) / P["Iz"]
    else:
        # force 모드: lon = 앞바퀴 종방향 힘 Fxf (바퀴 방향). 구동은 앞바퀴만(GTI 전륜),
        # 제동은 Fxf 가 전체의 brake_front_share 몫이라 보고 뒤바퀴 몫을 계산 (Stanford 데이터 생성 코드와 같은 배분)
        bb = cfg.brake_front_share
        Fxf = lon
        Fxr = torch.where(lon < 0, lon * (1.0 - bb) / bb, torch.zeros_like(lon))
        res = P["Cr0"] + P["Cr2"] * vx * vx
        c, s = torch.cos(delta), torch.sin(delta)
        dvx = (Fxf * c - Ffy * s + Fxr - res) / m + vy * r
        dvy = (Fry + Ffy * c + Fxf * s) / m - vx * r + P["kb"] * G * torch.sin(bank)
        dr = (lf * (Ffy * c + Fxf * s) - lr * Fry) / P["Iz"]
    if grade is not None:
        dvx = dvx - G * torch.sin(grade)
    return dvx, dvy, dr


def phys_step(cfg, s, u, P, tire=None):
    """s (B,3), u (B,3), P: 이름→(B,) 텐서. cfg.dt 를 cfg.nsub 번 나눈 오일러.
    tire=None 이면 Pacejka, 아니면 tire(af, ar, vx) → (Ffy, Fry)."""
    vx, vy, r = s.unbind(-1)
    vy = vy - P["ds"] * r  # 측정 위치의 vy → 무게중심 vy
    lon, delta, bank = u[..., 0], u[..., 1], u[..., 2]
    grade = u[..., 3] if u.shape[-1] > 3 else None
    h = cfg.dt / cfg.nsub
    for _ in range(cfg.nsub):
        af, ar = slips(cfg, vx, vy, r, delta)
        if tire is None:
            Ffy = pacejka(af, P["Bf"], P["Cf"], P["Df"], P["Ef"])
            Fry = pacejka(ar, P["Br"], P["Cr"], P["Dr"], P["Er"])
        else:
            Ffy, Fry = tire(af, ar, vx)
        dvx, dvy, dr = deriv(cfg, vx, vy, r, lon, delta, bank, Ffy, Fry, P, grade)
        vx, vy, r = vx + h * dvx, vy + h * dvy, r + h * dr
    return torch.stack([vx, vy + P["ds"] * r, r], -1)


def mlp(n_in, hidden, n_out, act=nn.Softplus):
    layers, d = [], n_in
    for hd in hidden:
        layers += [nn.Linear(d, hd), act()]
        d = hd
    layers.append(nn.Linear(d, n_out))
    net = nn.Sequential(*layers)
    for mm in net:
        if isinstance(mm, nn.Linear):
            nn.init.xavier_uniform_(mm.weight)
            nn.init.zeros_(mm.bias)
    return net


def _guard_buffers(cfg):
    lo = torch.tensor([p[2] for p in cfg.phys], dtype=torch.float32)
    hi = torch.tensor([p[3] for p in cfg.phys], dtype=torch.float32)
    init = torch.tensor([p[1] for p in cfg.phys], dtype=torch.float32)
    z = ((init - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)
    return lo, hi - lo, torch.log(z / (1 - z))


class GlobalPhys(nn.Module):
    """전역 계수 1벌. sigmoid 로 [하한, 상한] 안에 가둠 (학습 안정)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        lo, rng, z0 = _guard_buffers(cfg)
        self.register_buffer("lo", lo)
        self.register_buffer("rng", rng)
        self.raw = nn.Parameter(z0.clone())

    def values(self):
        return torch.sigmoid(self.raw) * self.rng + self.lo

    def P(self, B):
        v = self.values()
        return {k: v[i].expand(B) for i, k in enumerate(self.cfg.pnames)}


# ---------------------------------------------------------------- 사다리
class Base(nn.Module):
    uses_hist = True  # False 면 마지막 시점만 봄

    def __init__(self, cfg, norm):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        self.norm = norm


class R1(Base):
    uses_hist = False

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.g = GlobalPhys(cfg)

    def forward(self, hist):
        return phys_step(self.cfg, hist[:, -1, :3], hist[:, -1, 3:], self.g.P(len(hist)))


class R2(Base):
    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.g = GlobalPhys(cfg)
        self.res = mlp(cfg.nf * cfg.hist, [64, 64], 3)
        nn.init.zeros_(self.res[-1].weight)  # 시작은 R1 과 같음

    def forward(self, hist):
        base = phys_step(self.cfg, hist[:, -1, :3], hist[:, -1, 3:], self.g.P(len(hist)))
        return base + self.res(self.norm.x(hist)) * self.norm.dsd


class AxleNet(nn.Module):
    """α 에 대해 홀함수인 축 타이어 횡력. 입력 (α*10, vx/50). |F| <= Fmax (Fmax 도 학습)."""

    def __init__(self, fmax0):
        super().__init__()
        self.net = mlp(2, [32, 32], 1)
        self.logf = nn.Parameter(torch.tensor(math.log(fmax0)))

    def forward(self, a, vx):
        v = (vx / 50.0).unsqueeze(-1)
        g = lambda aa: self.net(torch.cat([10.0 * aa.unsqueeze(-1), v], -1)).squeeze(-1)
        return torch.exp(self.logf) * torch.tanh(0.5 * (g(a) - g(-a)))


class R3(Base):
    uses_hist = False

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.g = GlobalPhys(cfg)  # 구동·저항·Iz·kb·ds 만 씀 (타이어 계수는 안 씀)
        self.ft, self.rt = AxleNet(cfg.axle_fmax0[0]), AxleNet(cfg.axle_fmax0[1])

    def forward(self, hist):
        tire = lambda af, ar, vx: (self.ft(af, vx), self.rt(ar, vx))
        return phys_step(self.cfg, hist[:, -1, :3], hist[:, -1, 3:], self.g.P(len(hist)), tire=tire)


class R4(Base):
    """Deep Dynamics 식: 이력 → Physics Guard(sigmoid×범위+하한) → 샘플별 계수 → R1 과 같은 물리 식.
    DD 원본은 GRU 3층·과거 15스텝, 여기선 다른 단과 맞춰 MLP·과거 cfg.hist 스텝."""

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.body = mlp(cfg.nf * cfg.hist, [128, 128], len(cfg.phys))
        lo, rng, z0 = _guard_buffers(cfg)
        self.register_buffer("lo", lo)
        self.register_buffer("rng", rng)
        with torch.no_grad():  # 시작점 = 초기 물리값 (R1 시작과 같음)
            self.body[-1].bias.copy_(z0)

    def coeffs(self, hist):
        return torch.sigmoid(self.body(self.norm.x(hist))) * self.rng + self.lo

    def forward(self, hist):
        c = self.coeffs(hist)
        P = {k: c[:, i] for i, k in enumerate(self.cfg.pnames)}
        return phys_step(self.cfg, hist[:, -1, :3], hist[:, -1, 3:], P)


class R5(Base):
    """Spielberg(Stanford 2019) 식: MLP(이력) → 변화량(표준화), 다음 = 현재 + 변화량."""

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.net = mlp(cfg.nf * cfg.hist, [128, 128], 3)

    def forward(self, hist):
        return hist[:, -1, :3] + self.net(self.norm.x(hist)) * self.norm.dsd


class R6(Base):
    """구조 없는 블랙박스: MLP(이력) → 다음 상태 자체 (표준화 공간)."""

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.net = mlp(cfg.nf * cfg.hist, [128, 128], 3)

    def forward(self, hist):
        return self.net(self.norm.x(hist)) * self.norm.sd[:3] + self.norm.mu[:3]


RUNGS = dict(R1=R1, R2=R2, R3=R3, R4=R4, R5=R5, R6=R6)
RUNG_KO = dict(R1="순수 물리", R2="물리+잔차", R3="신경망 타이어", R4="DD식 계수추정", R5="Stanford식 NN", R6="블랙박스")


def build(rung, cfg, norm=None, device="cpu"):
    return RUNGS[rung](cfg, norm if norm is not None else Norm(nf=cfg.nf)).to(device)


def save_model(model, rung, path):
    torch.save(dict(rung=rung, cfg=model.cfg.name, state=model.state_dict()), path)


def load_model(path, cfg, device="cpu"):
    """cfg 는 저장할 때와 같은 VehicleConfig 여야 함 (계수 순서·범위가 state 에 맞아야 함)."""
    d = torch.load(path, map_location=device)
    m = build(d["rung"], cfg, Norm(nf=cfg.nf), device)
    m.load_state_dict(d["state"])
    return m.eval()


# ---------------------------------------------------------------- 여러 스텝 굴리기
@torch.no_grad()
def rollout(model, hist, U):
    """hist (B, H, 6) 에서 시작해 K = U.shape[1] + 1 스텝 굴림 → (B, K, 3) 예측 상태.
    첫 스텝 입력은 hist 마지막 행에 이미 있음. U (B, K-1, 3) 는 그다음 스텝들의 입력.
    출력[:, 0] = model(hist), 출력[:, k] = 예측 상태와 U[:, k-1] 을 이력 끝에 붙인 뒤의 예측.
    (학습의 연속 예측 손실, MPPI 후보 굴리기와 같은 방식)"""
    out = []
    for k in range(U.shape[1] + 1):
        s = model(hist)
        out.append(s)
        if k < U.shape[1]:
            hist = torch.cat([hist[:, 1:], torch.cat([s, U[:, k]], 1).unsqueeze(1)], 1)
    return torch.stack(out, 1)
