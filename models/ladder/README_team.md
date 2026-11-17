# 물리 비중 사다리 R1~R6 차량 모델 (팀 공유용)

KUBIG 물리×AI 차량 모델 · 2026-11-17 정리 · 김석우

R1~R6 차량 모델 코드만 따로 모은 폴더임. 데이터 읽기, 트랙, 시뮬레이터, MPPI 제어기는 들어 있지 않음.

## 이 모델은 데이터·트랙·시뮬레이터와 무관한가

| 무엇 | 무관한가 | 이유 |
|---|---|---|
| 트랙 | 무관 | 모델은 "지금 속도·회전 + 조향·구동 → 다음 순간 속도·회전"만 계산함. 트랙 모양을 모름 |
| 시뮬레이터 | 무관 | 시뮬레이터는 모델을 불러 쓰는 쪽임. 학습한 세계, Stanford 시뮬, 직접 짠 물리 시뮬 어디에 끼워도 같음 |
| 제어기 (MPPI 등) | 무관 | 제어기는 `rollout()`으로 후보 조작을 굴려 볼 뿐임 |
| 데이터 | 구조는 무관, 설정은 차마다 다름 | 차량 제원, 종방향 입력 종류(페달 / 힘), 시간 간격, 물리 계수 범위는 데이터·차량마다 정해야 함. 이 값만 `configs.py`에 모았음 |

- IAC(AV-21)와 Revs(Corvette) 실험은 원래 코드 두 벌(`src/iac_lib.py`, `src/rv_lib.py`)이었음
  - 두 파일의 모델 부분은 차량 값만 빼면 같았음
  - 그래서 한 파일(`ladder.py`)에 설정(`VehicleConfig`)만 갈아 끼우게 합쳤음

## 파일

| 파일 | 내용 |
|---|---|
| `ladder.py` | R1~R6 모델, 물리 식, 표준화, 저장·불러오기, 여러 스텝 굴리기(`rollout`) |
| `configs.py` | 차량 설정 3개: `AV21_IAC`, `CORVETTE_REVS`(지금까지 쓴 값 그대로), `GTI_THUNDERHILL`(초안) |
| `train.py` | 학습 함수 `fit()`, 창 만들기 `windows()`. 지금까지 실험과 같은 손실·옵티마이저 |
| `thunderhill_data.py` | 썬더힐 GTI 데이터 읽기 예시. 새 데이터를 붙일 때 이런 파일 하나만 새로 쓰면 됨 |
| `tests/check_same_as_repo.py` | 원본 코드와 출력이 같은지 확인 (우리 저장소 안에서만 돌아감) |
| `tests/smoke_thunderhill.py` | 썬더힐 데이터로 6개가 돌아가는지만 보는 짧은 시험 |

필요한 것: Python 3, `torch`, `numpy` (만든 환경: Python 3.14.7, torch 2.11.0, numpy 2.5.3)

## 사다리 (아래로 갈수록 물리가 적음)

| 단 | 이름 | 하는 일 | 과거 이력 사용 |
|---|---|---|---|
| R1 | 순수 물리 | 자전거 모델 + Pacejka 타이어 + 구동·저항 식. 계수는 데이터 전체에 1벌 | 마지막 시점만 |
| R2 | 물리+잔차 | R1 결과에 신경망 보정량을 더함 (시작은 보정 0) | 8시점 |
| R3 | 신경망 타이어 | 자전거 식 그대로, 앞·뒤 타이어 횡력만 신경망 (미끄럼각 부호를 바꾸면 힘 부호도 바뀌게 만듦) | 마지막 시점만 |
| R4 | DD식 계수추정 | 신경망이 순간마다 계수를 고르고(정해진 범위 안), R1과 같은 식에 넣음 | 8시점 |
| R5 | Stanford식 NN | 신경망이 변화량을 내고, 다음 = 현재 + 변화량 | 8시점 |
| R6 | 블랙박스 | 신경망이 다음 상태를 바로 냄 | 8시점 |

- 신경망 몸통은 R4·R5·R6이 128×128, R2 잔차가 64×64, R3 타이어가 32×32임. 활성화는 Softplus
- R1과 R4는 식·계수 목록이 같음. 차이는 "계수가 고정이냐, 순간마다 바뀌냐" 하나뿐임
- R4가 Deep Dynamics 원본과 다른 점
  - GRU → MLP, 과거 15 → 8스텝
  - 제동 이득 Cb 추가 (원본은 브레이크를 음수 스로틀로 구동 식에 넣음)
  - 경사 kb·센서 위치 ds 추가, 타이어 이동항 Sh·Sv 4개 제거
  - 계수 범위를 실제 차 크기로 넓힘
  - 한 스텝을 4번 나눠 적분

## 입력·출력 약속

- 상태 `s = [vx, vy, r]` (m/s, m/s, rad/s). vx는 앞(+), vy는 왼쪽(+), r은 반시계(+)
- 입력 `u = [lon, delta, bank]`
  - `lon`: 종방향 입력
    - `lon_input="pedal"`: 스로틀/브레이크 비율, −1~1, 브레이크는 음수
    - `lon_input="force"`: 앞바퀴 종방향 힘 Fxf (N), 제동은 음수
  - `delta`: 앞바퀴 조향각 (rad)
  - `bank`: 노면 횡경사 (rad). 없으면 0
- 모델 입력 `hist`: (배치, H, 6) = 과거 H 시점의 `[vx, vy, r, lon, delta, bank]`. 오래된 것이 앞, 마지막 행의 입력이 이번 스텝 입력
- 모델 출력: (배치, 3) = `dt` 초 뒤의 `[vx, vy, r]`
- 지금까지 실험 값은 `dt = 0.04`초(25 Hz), `H = 8`(0.32초)

## 쓰는 법

```python
import sys; sys.path.insert(0, "ladder_models")
import ladder
from configs import GTI_THUNDERHILL as CFG
from train import windows, fit
import thunderhill_data as th

d = "…/Model_Learning/data/exp/exp_data/dry"            # exp_*.npz 폴더
segs = {k: th.load_split(d, v) for k, v in th.EXAMPLE_SPLIT.items()}   # 주행 단위로 먼저 나눔
tr = windows(segs["train"], CFG.hist, horizon=1)        # 1스텝 학습 (논문식)
va = windows(segs["val"], CFG.hist, horizon=1)
model, rec = fit("R4", CFG, tr, va, epochs=300, seed=0)
ladder.save_model(model, "R4", "R4_s0.pt")

# 1초(25스텝) 굴려 보기: X (B,8,6), U (B,24,3) → (B,25,3)
X, Y, U = windows(segs["test"], CFG.hist, horizon=25)
import torch
pred = ladder.rollout(model, torch.tensor(X), torch.tensor(U))
```

- 1초 손실로 학습: `windows(..., horizon=25)`로 만든 창을 `fit()`에 넣으면 됨. `fit()`이 정답 길이를 보고 25스텝 굴리며 학습함
- 시뮬레이터나 MPPI에 넣을 때: 모델 하나 = "다음 0.04초 계산기"
  - 이력 8칸을 들고 다니며 `model(hist)`를 부르면 됨
  - 매 스텝 새 상태와 입력을 이력 끝에 붙임 (`rollout()`이 하는 일과 같음)

## 새 데이터·새 차를 붙이려면

1. **데이터 읽기 파일 하나**
   - 주행(파일)마다 `[(s (T,3), u (T,3)), ...]` 구간 목록을 만들면 됨. 저속(vx ≤ 5 m/s)과 끊긴 곳은 구간을 나눔
   - 학습·검증·시험은 반드시 주행 단위로 먼저 나눔
   - 창을 만든 뒤 섞어서 나누면 겹치는 창이 양쪽에 들어가 점수가 부풂. Stanford 원 코드가 이 문제를 가짐
2. **`configs.py`에 `VehicleConfig` 하나**
   - 제원: m, lf, lr
   - 종방향 입력 종류
   - dt, H
   - 계수 표 `(이름, 초기값, 하한, 상한)`
     - 필수 계수: Bf Cf Df Ef Br Cr Dr Er Cr0 Cr2 Iz kb ds
     - pedal 모드면 Cm1 Cm2 Cb도 필요함
   - 범위가 이상하면 `check()`가 바로 알려 줌
3. 모델 코드(`ladder.py`)는 건드리지 않음

## 썬더힐(GTI)에서 아직 정할 것

`GTI_THUNDERHILL`과 `thunderhill_data.py`는 초안임. 아래는 팀 결정이 필요함.

- **시간 간격**
  - 100 Hz 그대로(0.01초)로 갈지, 4개마다 뽑아 0.04초로 갈지 정해야 함. 초안은 다른 데이터·MPPI와 맞춘 0.04초
  - 공개 파일이 이미 저역 통과된 값인지는 확인 안 함
- **주행 나누기**
  - 11개 주행은 두 종류로 보임 [추정, 파일 값으로만 판단]
    - 빠른 코스 주행 5개: exp_2·8·9·10·16. vx 최대 약 33 m/s, 조향 ≤ 0.16 rad
    - 느리고 크게 꺾는 주행 6개: exp_4·5·6·11·12·15. vx ≤ 19 m/s, 조향 최대 0.26~0.47 rad. 스키드패드로 보임
  - `EXAMPLE_SPLIT`은 두 종류를 양쪽에 섞어 놓은 예시일 뿐임
- **종방향 힘 처리 (force 모드)**
  - 데이터에 페달이 없고 앞바퀴 힘 `fxf`만 있음
  - 구동은 앞바퀴만(GTI 전륜)으로 둠
  - 제동은 `fxf`가 전체 제동력의 66%라고 보고 뒤바퀴 몫을 계산함. Stanford 데이터 생성 코드의 배분을 따른 것임
  - 앞바퀴 힘이 조향 방향으로 작용하는 항(sin δ)을 넣었음
  - pedal 모드에 있던 구동 계수(Cm1·Cm2·Cb)는 없음. 저항(Cr0·Cr2)만 학습함
- **계수 초기값·범위**
  - 초기값은 Stanford 제원에서 가져옴
    - 정적 축하중으로 Df 9500·Dr 7900
    - 코너링 강성 150·170 kN/rad에 맞춰 Bf 12.1·Br 16.5
    - Iz 2763, 굴림저항 255 N, 공기저항 0.3638
  - 범위는 정해진 값이 아님. 학습해 보고 계수가 범위 끝에 붙으면 넓혀야 함
- **위치 정보 없음**: 데이터에 x, y가 없어서 썬더힐 코스 모양을 이 데이터로 만들 수 없음. 주행 시험 트랙은 따로 정해야 함

## 지금까지 실험에서 알게 된 주의점

- **1스텝 학습의 문제**: 물리 계열(R1~R4)을 1스텝으로만 학습하면 타이어 힘이 비현실적으로 작아질 수 있음
  - IAC R1: 최대 횡가속 약 0.34 g. 1초 손실로 학습하면 약 1.75 g
  - Revs에서는 거의 없었음 (1스텝 1.17 g vs 1초 손실 1.27 g)
  - 주행 시험이나 정답 세계로 쓸 모델은 1초 손실(`horizon=25`) 학습을 같이 볼 것
- **R4 계수가 튐**: 학습에 안 쓴 주행에서 계수가 범위 양 끝을 오갔음. 평균 반응은 비슷했지만 주행 시험에서 자주 이탈함
- **예측 1등 ≠ 주행 1등**: Revs에서 R5가 예측 오차로는 1등이었는데, 주행 시험에서는 가장 자주 실패함. 제동 반응을 작게 예측했기 때문임
- **결과 저장**: 마지막 에폭 모델을 씀. 검증 최저 에폭은 기록만 함

## 검증한 것 (2026-11-17)

- **원본과 같은 출력**
  - `tests/check_same_as_repo.py`로 원본 체크포인트(IAC·Revs, R1~R6, 시드 0)를 이 코드로 읽어 시험 주행 창에 넣어 봄
  - 1스텝 출력과 25스텝 굴린 출력이 원본 코드와 비트 단위로 같았음 (최대 차 0)
  - 대조: 질량만 790 → 800 kg으로 바꾸면 차이가 0.0025로 바로 나타남. 시험이 헛돌지 않음을 확인함
- **썬더힐에서 돌아감**
  - `tests/smoke_thunderhill.py`, 예시 분할, 각 3에폭: 6개 모두 손실이 유한하고 25스텝 굴리기도 유한했음
  - 3에폭짜리라 성능 비교에 쓰면 안 됨
- force 모드 식 자체가 썬더힐 차를 잘 설명하는지는 아직 확인 안 함. 300에폭 학습과 계수 점검이 다음 단계임

## 출처

- Deep Dynamics: Chrosniak 외, IEEE RA-L 2024, github.com/linklab-uva/deep-dynamics (R4, IAC 데이터)
- Stanford: Spielberg 외, Science Robotics 2019, purl.stanford.edu/zb950hd3384 (R5, 썬더힐 GTI 데이터·제원)
- Revs: Stanford Revs Program, purl.stanford.edu/tt103jr6546 (Corvette, ODC-By 1.0)
