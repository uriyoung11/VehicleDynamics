# 팀 공통 기준 모델 R1~R6 (팀원 사다리) — 우리 저장소 사본

팀 결정(11/17): 기준 모델은 팀원(김석우)의 사다리 R1~R6으로, MPPI는 팀원 구현으로, 계획 길이는 1초로 통일한다. 데이터는 **D7(Stanford 2025 Golf) 고정**.

- 원본 설명: [README_team.md](README_team.md) (팀원 `ladder_models.zip`, 2026-11-17)
- 모델·학습 코드(`ladder.py`, `train.py`, `configs.py`, `thunderhill_data.py`)는 원본을 그대로 가져왔고, 아래만 더했다.

## 우리가 더한 것

| 무엇 | 파일 | 내용 |
|---|---|---|
| 경사 채널 (선택) | `ladder.py`, `train.py` | `VehicleConfig.use_grade=True`면 입력이 7채널 `[vx, vy, r, lon, delta, bank, grade]`이 되고 dvx에 −g·sin(경사)를 더한다. **끄면 원본과 출력이 비트 단위로 같다** (`tests/check_same_as_team.py`, R1~R6 × pedal·force 설정, 1스텝·25스텝 차이 0) |
| Golf 설정 | `configs.py` | `GOLF_D7_FORCE`, `GOLF_D7_PEDAL`, `golf(lon, grade)`. 제원·초기값은 D7로 추정한 Golf 시뮬레이터 값(`physics/params.py` GOLF) |
| D7·D1 데이터 | `data/ladder_data.py` | `d7_segments(split, lon, road)`, `d1_segments(split, road)`. 0.04 s, 주행 단위 분할 |
| 시험 | `tests/smoke_golf.py` | Golf 조합별로 R1~R6이 도는지 (1스텝·1초 손실) |

## 비교할 조합 (팀 논의 11/17)

| 축 | 선택지 |
|---|---|
| 학습 데이터 | D7만 / D7 + D1 (D1은 GTI라 힘을 같은 가속도가 나오게 Golf 기준으로 환산. 위치·페달이 없어 노면 0, force만) |
| 종방향 입력 | force (앞바퀴 힘: 엔진 토크·기어비·브레이크 힘) / pedal (스로틀, −브레이크압) |
| 노면 | none / bank (팀원 방식: 좌우 기울기, 이득 kb 학습) / bank+grade (우리 방식: + 오르막·내리막) |
| 손실 | 1스텝 / 1초(25스텝) |

## 확인한 것 (11/17)

- **환산이 맞다:** 학습 전 R1(초기값 = Golf 시뮬레이터 추정값)의 D7 시험 1초 롤아웃 RMSE가 vx 0.460 / vy 0.361 / r 0.047로, 우리 Golf 시뮬레이터(0.433 / 0.360 / 0.048)와 거의 같다.
- **경사 효과:** 같은 R1에 경사를 넣으면 vx 오차가 0.460 → 0.386 (약 16% 감소). 시뮬레이터에서 본 효과와 같다.
- 좌우 기울기만(팀원 방식)은 kb 초기값이 0이라 학습 전에는 노면 없음과 같다.
- pedal 모드의 구동·제동 계수(Cm1, Cm2, Cb) 초기값은 대략값이라 학습 전 vx 오차가 크다(1.19 m/s).
- 데이터: D7 학습 28.0분 / 검증 / 시험 5.2분, D1 학습 15.3분 / 시험 2.6분 (25 Hz).

## 쓰는 법

```python
import sys; sys.path[:0] = ["models/ladder", "."]
import ladder, configs
from train import windows, fit
from data.ladder_data import d7_segments, d1_segments

cfg = configs.golf("force", grade=True)                      # 우리 방식 노면
tr = d7_segments("train", "force", "bank+grade") + d1_segments("train", "bank+grade")   # D7 + D1
va = d7_segments("val", "force", "bank+grade")
model, rec = fit("R2", cfg, windows(tr, cfg.hist, horizon=25), windows(va, cfg.hist, horizon=25), epochs=300)
```
