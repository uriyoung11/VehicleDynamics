"""차량·타이어 파라미터와 프리셋.

필드는 float 또는 (N,) 텐서 둘 다 받는다. 에피소드마다 질량·강성을 흔들 때 텐서로 넣으면
모든 식이 브로드캐스팅으로 그대로 돌아간다.
"""
from dataclasses import dataclass, field, replace
import math


@dataclass
class TireParams:
    # 하중으로 정규화한 코너링 강성 C_alpha / Fz [1/rad]. 마찰 μ가 바뀌어도 이 값은 유지되고,
    # 대신 포화 시작 슬립각이 작아진다(Fiala 모델과 같은 성질).
    c_alpha: float
    C: float = 1.6   # Pacejka 형상 계수. 2를 넘으면 큰 슬립에서 횡력 부호가 뒤집힌다
    E: float = 0.0   # Pacejka 곡률 계수. peak_slip 해석식은 E=0에서만 쓴다


@dataclass
class VehicleParams:
    name: str
    m: float            # 질량 [kg]
    Iz: float           # 요 관성 [kg m^2]
    a: float            # 무게중심 ~ 앞 차축 [m]
    b: float            # 무게중심 ~ 뒤 차축 [m]
    h: float            # 무게중심 높이 [m]
    tire_f: TireParams
    tire_r: TireParams
    delta_max: float        # 최대 조향각 [rad]
    delta_rate_max: float   # 최대 조향 각속도 [rad/s]
    drive_front: float      # 구동력 중 앞바퀴 비율
    brake_front: float      # 제동력 중 앞바퀴 비율
    power_max: float        # 최대 구동 출력 [W]
    c0: float = 0.0         # 주행 저항 F = c0 + c1*Ux + c2*Ux^2 [N]
    c1: float = 0.0
    c2: float = 0.0
    g: float = 9.81
    # 생성기 설정
    ux_min_valid: float = 2.0                     # 이보다 느리면 에피소드 종료 [m/s]
    ux_target_range: tuple = (5.0, 30.0)          # 목표 속도 범위 [m/s]
    noise_std: dict = field(default_factory=dict)  # 센서 잡음 표준편차

    @property
    def L(self):
        return self.a + self.b

    def scaled(self, mass=1.0, stiffness=1.0):
        """질량(관성 포함)과 코너링 강성 배율을 적용한 복사본."""
        return replace(self, m=self.m * mass, Iz=self.Iz * mass,
                       tire_f=replace(self.tire_f, c_alpha=self.tire_f.c_alpha * stiffness),
                       tire_r=replace(self.tire_r, c_alpha=self.tire_r.c_alpha * stiffness))


def _c_alpha_norm(C_alpha, m, a, b, axle, g=9.81):
    """[N/rad] 코너링 강성을 정적 하중으로 나눠 정규화."""
    L = a + b
    fz = m * g * (b if axle == "f" else a) / L
    return C_alpha / fz


# 풀스케일: Spielberg 2019 VW GTI (learning_params.py). 전륜구동, 제동 배분 0.66.
# h, 조향 각속도, 주행 저항은 가정값.
_m, _a, _b = 1776.2, 1.194, 1.437
FULLSCALE = VehicleParams(
    name="fullscale_gti",
    m=_m, Iz=2763.49, a=_a, b=_b, h=0.55,
    tire_f=TireParams(c_alpha=_c_alpha_norm(150.0e3, _m, _a, _b, "f")),   # ≈15.8 /rad
    tire_r=TireParams(c_alpha=_c_alpha_norm(170.0e3, _m, _a, _b, "r")),   # ≈21.5 /rad
    delta_max=math.radians(27.0), delta_rate_max=1.0,
    drive_front=1.0, brake_front=0.66, power_max=147e3,
    c0=0.015 * _m * 9.81, c1=0.0, c2=0.43,   # 구름저항 1.5%, 0.5*rho*CdA (CdA≈0.7) — 가정
    ux_min_valid=2.0, ux_target_range=(5.0, 30.0),
    noise_std=dict(ux=0.05, uy=0.02, r=0.005, ax=0.1, ay=0.1),
)

# 소형: D2 F1TENTH (Xray GTXE'22, 4WD). 저장소 single_track_parameters.py / pacejka.py 초기값 기준.
# 코너링 강성 = B*C*D = 5.948*2.275*0.8095 ≈ 10.95 /rad. h, 조향 각속도, 출력은 가정값.
F1TENTH = VehicleParams(
    name="f1tenth_xray",
    m=5.1, Iz=0.46, a=0.33 - 0.115, b=0.115, h=0.04,
    tire_f=TireParams(c_alpha=10.95), tire_r=TireParams(c_alpha=10.95),
    delta_max=0.531, delta_rate_max=6.0,
    drive_front=0.115 / 0.33, brake_front=0.115 / 0.33, power_max=400.0,
    c0=0.1, c1=0.01, c2=0.1,
    ux_min_valid=0.5, ux_target_range=(1.5, 6.0),
    noise_std=dict(ux=0.02, uy=0.02, r=0.02, ax=0.3, ay=0.3),
)

# [11/19] 이 GOLF 프리셋(우리 Golf 시뮬레이터, m 1630)은 쓰지 않는다. 팀 기준 차량·시뮬레이터는 팀원 것 (docs/status_2026-11-19.md).
# 풀스케일 Golf (D7 데이터상 이름 "VW Golf", 엔진·LSD·구동 단서로 GTI로 추정): D7(Stanford 2025, Thunderhill West)으로 추정
# (physics/golf_sysid.py --fix a=1.05 --slope, physics/golf_fit_slope.json).
# 공개 제원이 없어 고정한 값: 질량 1630 kg(브레이크 힘-감속도 관계로 추정), 축거 2.63 m(제조사),
#   a = 1.05 m(제조사 앞 무게 약 60%. 데이터로는 정해지지 않음), 전륜구동, 앞 제동 배분 0.75(D7 측정).
# 데이터로 맞춘 값: Iz, 정규화 코너링 강성·Pacejka 형상(앞뒤), μ, h, 주행 저항. 0.5 s 롤아웃 손실.
# 노면 경사를 넣고 맞췄다(트랙 파일 data/tracks/ThunderhillWest_centerline.csv의 grade_rad 열, rk4_step(..., slope=)).
#   D7 시험 주행 2 s 롤아웃 오차: r 0.051 rad/s, Uy 0.41 m/s, Ux 0.70 m/s (평지로 맞춘 값 0.052, 0.42, 0.82;
#   GTI 제원 그대로 0.219, 3.34, 2.63). 0.8 g 이상 구간에서도 r 0.060, Uy 0.49.
# 이 프리셋과 함께 쓸 노면 μ는 1.085. D7의 Vy는 무게중심보다 약 0.27 m 앞에서 잰 값이다(Vy_측정 = Vy_무게중심 + 0.27·r).
# 좌우 기울기: 맞춘 이득이 −0.23(측정 기울기의 1/4, 반대 부호)이라 효과가 작고 믿기 어렵다. 기본은 경사만 쓴다.
#   기울기도 넣으려면 slope = [grade, GOLF_BANK_GAIN·bank].
# Iz가 크게 나온 것은 타이어 반응 지연 등 모델에 없는 효과를 흡수했을 수 있다. c2도 일반 Golf(약 0.4)보다 크다.
_mg = 1630.0
GOLF = VehicleParams(
    name="golf_d7",
    m=_mg, Iz=3630.2, a=1.05, b=1.58, h=0.319,
    tire_f=TireParams(c_alpha=11.815, C=1.301), tire_r=TireParams(c_alpha=16.631, C=1.815),
    delta_max=0.61, delta_rate_max=1.0,          # D7 최대 조향 0.61 rad. 조향 속도는 99.9분위 0.63 rad/s라 여유 있게 1.0
    drive_front=1.0, brake_front=0.75, power_max=150e3,   # D7 바퀴 구동 출력 99.9분위 148 kW
    c0=24.9, c1=0.0, c2=1.258,
    ux_min_valid=2.0, ux_target_range=(5.0, 40.0),
    noise_std=dict(ux=0.05, uy=0.02, r=0.005, ax=0.1, ay=0.1),
)
GOLF_MU = 1.085
GOLF_VY_SENSOR_OFFSET = 0.27   # [m]
GOLF_BANK_GAIN = -0.233        # 좌우 기울기 이득 (믿기 어려움, 기본은 쓰지 않음)

PRESETS = {"fullscale": FULLSCALE, "f1tenth": F1TENTH, "golf": GOLF}
