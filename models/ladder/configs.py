"""차량별 설정. 모델 구조는 ladder.py 하나이고, 차가 바뀌면 여기 값만 바뀜.

AV21_IAC, CORVETTE_REVS : 지금까지 실험에 쓴 값 그대로 (src/iac_lib.py, src/rv_lib.py 에서 옮김)
GTI_THUNDERHILL          : 썬더힐 Stanford VW GTI 용 초안. 아직 학습·검증 안 함
"""
from ladder import VehicleConfig

# ------------------------------------------------------------ IAC AV-21 (Deep Dynamics 공개 데이터)
# 제원: DD 설정 파일 deep_dynamics_iac.yaml. 범위: DD iac 설정을 바탕으로 넓히고 Cb·kb·ds 추가
AV21_IAC = VehicleConfig(
    name="AV21_IAC", m=790.0, lf=1.248, lr=1.7328, dt=0.04, hist=8, lon_input="pedal",
    axle_fmax0=(6000.0, 7000.0),
    phys=[  # 이름, 초기값, 하한, 상한
        ("Bf", 12.0, 5.0, 30.0), ("Cf", 1.3, 0.5, 2.0), ("Df", 5000.0, 100.0, 12000.0), ("Ef", -0.5, -2.0, 0.0),
        ("Br", 12.0, 5.0, 30.0), ("Cr", 1.3, 0.5, 2.0), ("Dr", 6000.0, 100.0, 12000.0), ("Er", -0.5, -2.0, 0.0),
        ("Cm1", 6000.0, 100.0, 15000.0), ("Cm2", 20.0, 0.0, 200.0), ("Cb", 8000.0, 100.0, 20000.0),
        ("Cr0", 200.0, 0.0, 2000.0), ("Cr2", 0.5, 0.0, 3.0), ("Iz", 1000.0, 500.0, 2000.0),
        ("kb", 0.0, -1.5, 1.5),   # 뱅크 중력 성분 이득: dvy += kb * g * sin(bank)
        ("ds", 0.0, -3.0, 3.0)],  # 측정 vy 위치 보정 (m): vy_측정 = vy_무게중심 + ds * r
    note="lon = 스로틀/100, 브레이크 눌리면 -브레이크압/2757.9. bank = LVMS roll 열(1초 평활), Putnam 0",
)

# ------------------------------------------------------------ Revs 1963 Corvette Grand Sport (Laguna Seca)
# 제원: Revs 제원 파일 grandsport.csv (1014 kg, a 1.2 m, b 1.29 m). 범위는 1014 kg 차에 맞춰 조정
CORVETTE_REVS = VehicleConfig(
    name="CORVETTE_REVS", m=1014.0, lf=1.2, lr=1.29, dt=0.04, hist=8, lon_input="pedal",
    axle_fmax0=(6000.0, 6000.0),
    phys=[
        ("Bf", 12.0, 3.0, 30.0), ("Cf", 1.3, 0.5, 2.0), ("Df", 5000.0, 100.0, 15000.0), ("Ef", -0.5, -2.0, 0.0),
        ("Br", 12.0, 3.0, 30.0), ("Cr", 1.3, 0.5, 2.0), ("Dr", 5000.0, 100.0, 15000.0), ("Er", -0.5, -2.0, 0.0),
        ("Cm1", 5000.0, 100.0, 15000.0), ("Cm2", 50.0, 0.0, 300.0), ("Cb", 10000.0, 100.0, 30000.0),
        ("Cr0", 200.0, 0.0, 2000.0), ("Cr2", 0.5, 0.0, 3.0), ("Iz", 1500.0, 600.0, 3500.0),
        ("kb", 0.0, -1.5, 1.5),
        ("ds", 0.0, -1.0, 1.0)],
    note="lon = 스로틀/100, 브레이크 5% 넘으면 -브레이크/100. bank = 0. delta = 핸들각 표로 바꾼 앞바퀴각",
)

# ------------------------------------------------------------ Stanford VW GTI "Niki" (Thunderhill) — 초안
# 제원: Stanford learning_params.py / vehicles.py (m 1776.2, a 1.194, b 1.437, Izz 2763.49,
#       코너링 강성 Cf 150 kN/rad, Cr 170 kN/rad, 굴림저항 255 N, 공기저항 0.3638 N/(m/s)^2, 제동 앞 몫 0.66)
# 초기값: 정적 축하중 앞 9517 N, 뒤 7909 N (μ=1 이면 최대 횡력) → Df 9500, Dr 7900.
#         B 는 B·C·D 가 Stanford 코너링 강성과 맞게: 12.1·1.3·9500 ≈ 150 k, 16.5·1.3·7900 ≈ 170 k
# 범위: D 상한 25 kN (μ 약 2.6 까지 허용), Iz 1000~5000. 정해진 값 아님 — 학습해 보고 범위 끝에 붙으면 조정
# 데이터에 페달이 없고 앞바퀴 힘 fxf(N)가 있어서 lon_input="force"
GTI_THUNDERHILL = VehicleConfig(
    name="GTI_THUNDERHILL", m=1776.2, lf=1.194, lr=1.437, dt=0.04, hist=8, lon_input="force",
    brake_front_share=0.66, axle_fmax0=(9500.0, 7900.0),
    phys=[
        ("Bf", 12.1, 3.0, 30.0), ("Cf", 1.3, 0.5, 2.0), ("Df", 9500.0, 100.0, 25000.0), ("Ef", -0.5, -2.0, 0.0),
        ("Br", 16.5, 3.0, 30.0), ("Cr", 1.3, 0.5, 2.0), ("Dr", 7900.0, 100.0, 25000.0), ("Er", -0.5, -2.0, 0.0),
        ("Cr0", 255.0, 0.0, 2000.0), ("Cr2", 0.3638, 0.0, 3.0), ("Iz", 2763.49, 1000.0, 5000.0),
        ("kb", 0.0, -1.5, 1.5),   # 썬더힐 데이터에는 경사 열이 없어 bank=0 → 효과 없음 (형식 유지)
        ("ds", 0.0, -1.0, 1.0)],
    note="초안. lon = fxf (N). dt 0.04 는 100 Hz 를 4개마다 뽑는다는 가정 (thunderhill_data.py)",
)

# ------------------------------------------------------------ [우리 추가] D7 Stanford 2025 VW Golf (GTI로 추정), Thunderhill West
# 팀 결정(11/17): 데이터는 D7 고정. 데이터 읽기: data/ladder_data.py (d7_segments, d1_segments)
# 제원: 공개 자료 없음 → D7로 추정한 Golf 시뮬레이터 값 (physics/params.py GOLF, physics/golf_fit_slope.json)
#   m 1630 kg(브레이크 힘-감속도), lf 1.05 / lr 1.58 m(축거 2.63 m 제조사, 앞 무게 약 60%), 제동 앞 몫 0.75(D7 측정)
# 초기값: 우리 추정 타이어(정규화 코너링 강성 앞 11.8·뒤 16.6 /rad, 형상 C 1.30·1.82, μ 1.085)를 Pacejka B·C·D로 환산
#   정적 축하중 앞 9606 N·뒤 6384 N → Df = μ·Fz ≈ 10420, Dr ≈ 6930. B = 강성/(C·D) → Bf ≈ 8.4, Br ≈ 8.5. E 는 우리 모델이 0 → −0.1
#   Iz 3630, 굴림저항 25 N, 공기저항 1.26, 센서 위치 ds 0.27 m
# 범위는 GTI_THUNDERHILL 초안과 같은 폭(D 상한 25 kN 등). 학습해 보고 범위 끝에 붙으면 넓힌다.
_GOLF_TIRE_DRAG = [
    ("Bf", 8.4, 3.0, 30.0), ("Cf", 1.30, 0.5, 2.0), ("Df", 10420.0, 100.0, 25000.0), ("Ef", -0.1, -2.0, 0.0),
    ("Br", 8.5, 3.0, 30.0), ("Cr", 1.82, 0.5, 2.0), ("Dr", 6930.0, 100.0, 25000.0), ("Er", -0.1, -2.0, 0.0),
    ("Cr0", 25.0, 0.0, 2000.0), ("Cr2", 1.26, 0.0, 3.0), ("Iz", 3630.0, 1000.0, 5000.0),
    ("kb", 0.0, -1.5, 1.5),   # 좌우 기울기 이득 (road="bank"일 때 의미 있음)
    ("ds", 0.27, -1.0, 1.0)]
GOLF_D7_FORCE = VehicleConfig(
    name="GOLF_D7_FORCE", m=1630.0, lf=1.05, lr=1.58, dt=0.04, hist=8, lon_input="force",
    brake_front_share=0.75, axle_fmax0=(10420.0, 6930.0), phys=_GOLF_TIRE_DRAG,
    note="lon = 앞바퀴 힘 Fxf [N] (엔진 토크·기어비·브레이크 힘으로 계산). data.ladder_data.d7_segments(lon='force')",
)
# pedal: Fx = (Cm1 − Cm2·vx)·스로틀 + Cb·(−브레이크압/50) − 저항. 초기값은 D7 대략값:
#   Cb: 브레이크 49 bar에서 바퀴 브레이크 힘 합 약 13.6 kN → lon −0.98 → Cb ≈ 13900
#   Cm1·Cm2: 엔진 출력 150 kW급, 저속 최대 구동력 약 6~7 kN 가정
GOLF_D7_PEDAL = VehicleConfig(
    name="GOLF_D7_PEDAL", m=1630.0, lf=1.05, lr=1.58, dt=0.04, hist=8, lon_input="pedal",
    axle_fmax0=(10420.0, 6930.0),
    phys=_GOLF_TIRE_DRAG + [("Cm1", 6500.0, 100.0, 20000.0), ("Cm2", 80.0, 0.0, 400.0), ("Cb", 13900.0, 100.0, 30000.0)],
    note="lon = 스로틀/100, 제동 중 −브레이크압/50 bar. data.ladder_data.d7_segments(lon='pedal')",
)


def golf(lon="force", grade=False):
    """lon: 'force' | 'pedal', grade: True 면 경사 채널(우리 방식) 사용 → use_grade=True, 이름에 _G."""
    from dataclasses import replace
    c = GOLF_D7_FORCE if lon == "force" else GOLF_D7_PEDAL
    return replace(c, use_grade=True, name=c.name + "_G") if grade else c


ALL = {c.name: c for c in (AV21_IAC, CORVETTE_REVS, GTI_THUNDERHILL, GOLF_D7_FORCE, GOLF_D7_PEDAL,
                           golf("force", True), golf("pedal", True))}
