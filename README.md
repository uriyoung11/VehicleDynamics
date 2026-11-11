# VehicleDynamics

KUBIG 프로젝트 "신경망 차량 모델로 한계 주행하기". 전체 계획은 [project.md](project.md), 진행 기록은 [docs/](docs/)에 있다.

## 폴더 구조

| 폴더 | 내용 |
|---|---|
| [physics/](physics/) | 미분 가능한 자전거 모델(P0~P2), 타이어 모델, 차량 프리셋(`fullscale` GTI, `f1tenth` Xray), CommonRoad 검증 |
| [data/](data/) | 데이터 로더(D1, D2, D3)와 합성 데이터 생성기. 데이터 파일 자체는 git에 올리지 않는다 |
| [models/](models/) | 기준 모델 M0(신경망), B1(피팅한 물리 모델)과 학습 스크립트 |
| [eval/](eval/) | 공통 평가 코드(1스텝, 0.5초·1초 롤아웃)와 결과 JSON |
| [repro/spielberg2019/](repro/spielberg2019/) | 원논문(Spielberg et al. 2019) PyTorch 재현 |
| [viz/](viz/) | 시각화. `demo_track.py`(matplotlib), `demo_rerun.py`(Rerun, 팀 분석용), `replay.py`(공용 장면 파일), `tools/`(Blender 발표 영상 등) |
| [tests/](tests/) | 물리 모델·생성기 검증 (`python tests/test_physics.py`) |
| [docs/](docs/) | 결과 정리 문서와 그림 |

## git에 없는 폴더 (각자 로컬)

| 폴더 | 내용 | 만드는 법 |
|---|---|---|
| `data/scirob_submission/` | D1 원논문 데이터·코드 | [Stanford 저장소](https://purl.stanford.edu/zb950hd3384)에서 받기 |
| `data/d2/` | D2 CSV 4개 | [D2 저장소](https://github.com/SzalonyJohny/Learning-dynamics-models-for-velocity-estimation)에서 받기 |
| `data/generated/` | D3 합성 데이터 (`*_v3_trial`) | `python -m data.synth --vehicle f1tenth --out data/generated/...` |
| `data/tracks/` | 트랙 중심선 | 시각화 처음 실행 때 자동으로 받음 |
| `checkpoints/` | 학습한 모델 | `python -m models.train` |
| `logs/` | 학습·생성 로그 | |
| `viz/out/` | 영상, Rerun 기록, 캐시 | 각 시각화 스크립트 실행 |

## 자주 쓰는 명령

```
python tests/test_physics.py                                      # 물리 검증
python -m models.train --data d3:data/generated/f1tenth_v3_trial --model m0
python -m eval.evaluate --data d1 --split test --ckpt "checkpoints/d1/*.pt"   # 평가
python -m viz.demo_rerun                                           # Rerun 분석 화면
python -m viz.replay                                               # 공용 장면 파일 만들기
blender -b -P viz/tools/blender_demo.py -- --fps 30                # 발표 영상 렌더링 (Blender 4.2 이상)
python -m viz.tools.blender_compose                                # 렌더링 결과 합성 -> viz/out/tools/blender.mp4
```
