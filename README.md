# VehicleDynamics

KUBIG 프로젝트 "신경망 차량 모델로 한계 주행하기" (Team 물리와 AI). **현재 기준과 팀 결정은 [docs/status_2026-11-19.md](docs/status_2026-11-19.md)**.
팀 공용 저장소는 팀원 저장소(physics_for_driving)이고, 이 저장소는 내 작업 기록이다. 이전 방향의 코드·문서는 [archive/](archive/)에 있다.

## 폴더 구조

| 폴더 | 내용 |
|---|---|
| [data/](data/) | D7(Golf)·D1(GTI) 로더(`datasets.py`, 사다리 형식 `ladder_data.py`), 트랙 만들기(`thunderhill_west.py` 중앙선·노면 지도, `thunderhill_track_v2.py` 트랙 npz), 팀원 분할(`d7_split_team.json`). 데이터 파일은 git에 올리지 않는다 |
| [models/ladder/](models/ladder/) | 팀 공통 기준 모델 R1~R6(팀원 사다리) + 경사 채널, 비교 실험·차량 간 평가 |
| [physics/](physics/) | 자전거 모델·타이어·차량 프리셋 (데이터 로더가 씀). 시뮬레이터로는 쓰지 않는다 |
| [viz/](viz/) | `ladder_predict.py`: 실제 주행을 재생하며 모델의 1초 예측선을 그리는 영상 |
| [eval/results/](eval/results/) | 사다리 비교 결과 JSON (11/19 이전 분할) |
| [docs/](docs/) | 현재 상태, 트랙·시뮬레이터 문서, 그림 |
| [archive/](archive/) | 이전 모델(M0~M3, B1)·MPPI·Golf 시뮬레이터·시각화·문서 |

## git에 없는 폴더 (각자 로컬)

| 폴더 | 내용 | 만드는 법 |
|---|---|---|
| `data/d7/` | D7 Golf CSV 22개 | [Stanford 저장소](https://purl.stanford.edu/hh613qz0317)에서 받기 |
| `data/scirob_submission/` | D1 GTI 데이터 (차량 간 평가용) | [Stanford 저장소](https://purl.stanford.edu/zb950hd3384)에서 받기 |
| `data/tracks/` | OSM 캐시, 트랙 v3 | 아래 명령 (OSM은 처음 실행 때 자동으로 받음) |
| `checkpoints/ladder/` | 사다리 모델 | `python models/ladder/run_ablation.py train ...` |
| `viz/out/` | 영상 | 각 시각화 스크립트 |

## 자주 쓰는 명령

```
# 트랙 v3 (학습 파일로만, 팀원 분할)
python -m data.thunderhill_west --split-json data/d7_split_team.json --out data/tracks/v3/ThunderhillWest_centerline_train.csv
python -m data.thunderhill_track_v2 --center-csv data/tracks/v3/ThunderhillWest_centerline_train.csv --split-json data/d7_split_team.json --out data/tracks/v3
python -m data.thunderhill_track_v2 --center-csv data/tracks/v3/ThunderhillWest_centerline_train.csv --split-json data/d7_split_team.json --width-points data/tracks/thunderhill_width_points.csv --out data/tracks/v3_local

# 사다리 비교 실험, 차량 간 평가
python models/ladder/run_ablation.py eval
python models/ladder/cross_eval.py

# 예측 영상
python viz/ladder_predict.py --models R1 R2 R4 R5
```
