# 신경망 차량 모델로 한계 주행하기 — 프로젝트 개요 보고서

KUBIG · Team 물리와 AI · 2026-11-05 (11/17 현재 상태 반영)

## 현재 상태 (11/17)

이 문서는 11/5 계획이다. 그 뒤 바뀐 점은 아래와 같고, 자세한 내용은 각 문서에 있다.

| 항목 | 11/5 계획 | 지금 |
|---|---|---|
| 학습 모델 | M0~M3, B1, B2(GP) | M0~M3, B1 구현 완료 ([docs/models_m1_m3.md](docs/models_m1_m3.md)). B2(GP)는 미구현. Deep Dynamics식 모델은 비교 후보로 검토 중 |
| 실데이터 | D1, D2 주력, D4·D5 보조 | D1(GTI) + **D7(Stanford 2025 Golf, 같은 Thunderhill)** 을 풀스케일 한계 실데이터로, D2는 실제 μ 변화 검증용. D4는 쓰지 않음. D5(AV-21)는 한계 주행 데이터가 아님 ([docs/dataset_survey_2026-11-12.md](docs/dataset_survey_2026-11-12.md), [docs/deep_dynamics_analysis_2026-11-12.md](docs/deep_dynamics_analysis_2026-11-12.md)) |
| 제어 (RQ5) | MPC 폐루프 경로 추종 | MPPI 구현. 방식 A(목표 속도 추종)와 B(진행 최대화) 비교 중. 정답 μ MPC도 실패할 만큼 MPPI가 불안정해 안정화가 먼저 ([docs/mpc_comparison_design_2026-11-16.md](docs/mpc_comparison_design_2026-11-16.md)) |
| 진행 중 | | 여러 차 데이터 혼합 학습(GTI + Golf)과 모델 입출력 형식 비교 ([docs/mixing_io_format_2026-11-17.md](docs/mixing_io_format_2026-11-17.md)) |
| 계산 자원 | GPU 예산 | 노트북 CPU로 학습. MPPI 대량 주행은 코랩 GPU 사용 예정 |

## 요약

남은 8주 동안 "차량 동역학 모델에 물리를 어디까지 넣어야 한계 주행 구간에서 이득인가"를 하나의 질문으로 정하고, 직접 만든 물리 모델 계층과 학습 모델 4종을 같은 평가 틀에서 비교한다.

- **연구 질문**: 순수 MLP, 물리 잔차 학습, 타이어 힘만 학습, 물리 손실(PINN) 중 어느 구조가 타이어 포화 구간의 멀티스텝 예측과 경로 추종에서 가장 나은가.
- **접근**: 자전거 모델부터 Pacejka 타이어, 하중 이동, 4륜 모델까지 물리 모델을 팀이 직접 구현한다. 이 시뮬레이터로 노면 μ를 바꾼 데이터를 만들고, 실데이터(Spielberg 공개 데이터, F1TENTH 데이터)로 검증한다.
- **평가**: 1스텝 MSE에 더해 멀티스텝 롤아웃 오차, 슬립각 구간별 오차, 불확실성 보정, MPC 폐루프 경로 추종을 본다.
- **산출물**: 비교 결과 보고서와 발표, 재현 가능한 코드 저장소, 롤아웃·주행 시각화 데모.

중간 발표의 실험 계획 4개(물리식 교체, 모델 비틀기, 노면·극한 상황, 데이터·시뮬레이션)는 모두 이 틀 안의 실험 축으로 들어간다.

## 배경과 관련 연구

원논문 이후 연구는 블랙박스 신경망에서 물리 구조 안에 학습 요소를 끼워 넣는 쪽으로 이동했고, 우리 질문은 그 흐름을 작은 규모로 비교하는 것이다.

| 계열 | 대표 연구 | 핵심 내용 | 우리 프로젝트에서의 역할 |
| --- | --- | --- | --- |
| 원논문 | [Spielberg et al., Science Robotics 2019](https://purl.stanford.edu/zb950hd3384) | 신경망 차량 모델을 FF-FB 경로 추종에 사용. 공개 자료는 고마찰 노면 실험 데이터, 차량 모델 데이터 생성 코드, 학습 코드 | 재현 기준선, 실데이터 1 |
| 원논문 후속 | [Spielberg 박사논문](https://purl.stanford.edu/wp128xy5795) | 신경망 MPC, 학습 영역 밖에서도 확신하는 예측의 문제와 불확실성 학습 | 불확실성 평가 축의 근거 |
| 신경망 타이어 모델 | [Broadbent et al., AVEC '24](https://arxiv.org/abs/2407.13760) | 전륜 횡력만 신경망으로 학습해 브러시 모델 자리에 대체, 드리프트 NMPC | "타이어 힘만 학습" 구조의 근거 |
| 신경망 타이어 모델 | [Neural-ExpTanh / NODE](https://arxiv.org/html/2306.06330v2) | 단일 트랙 모델 안의 타이어 모델을 학습, 기존 NMPC에 그대로 삽입 | 같은 구조의 다른 매개화 |
| 온라인 적응 | [First, Learn What You Don't Know](https://arxiv.org/pdf/2411.00107) | 물리 사전지식 + 신경망 특징, Bayesian last-layer로 빠른 적응 | 불확실성·적응 확장 후보 |
| 온라인 적응 | [TRI, Drifting with Unknown Tires](https://www.tri.global/research/drifting-unknown-tires-learning-vehicle-models-online-neural-networks-and-model-predictive) | NMPC 안 신경망 모델을 온라인 갱신, 타이어 교체 후 적응 | 노면 변화 실험의 동기 |
| PINN | [Chalmers 학위논문](https://odr.chalmers.se/bitstreams/ab7f406a-79f1-43a9-8d35-0bc5de66de0c/download), [FTHD](https://www.arxiv.org/pdf/2409.19647) | 단일 트랙 + Pacejka를 손실이나 구조에 넣은 PINN | "물리 손실" 구조의 근거 |
| GP 잔차 | [BayesRace](https://arxiv.org/pdf/2005.04755), [Ning & Behl](https://arxiv.org/abs/2306.03405v1) | 물리 모델과 실제의 불일치를 GP로 학습, F1TENTH 스케일 분석 | 비신경망 baseline |
| 스케일카 데이터 | [Węgrzynowski et al.](https://arxiv.org/abs/2408.15610v1) | 공격적 기동 데이터, 횡활각 40°까지, 타이어 4종 세트 | 실데이터 2 (한계 구간) |

**중간 발표까지의 결과**: 자전거 모델 변수를 현재와 과거 3시점까지 넣은 128×2 MLP로 원논문을 재현했다. 주행 단위로 분할해 누수를 고친 뒤 1스텝 MSE는 신경망 8.3, 물리 모델 44.0(×10⁻⁶)이다. 남은 한계는 세 가지로, 마른 노면 데이터뿐이고, 1스텝 지표뿐이며, 물리 baseline의 피팅 방식이 명시되지 않았다.

원논문 본문 대조 전이므로 "신경망이 물리 모델보다 훨씬 낫다"는 표현은 피한다. 보도 기준으로 실차에서는 두 모델이 비슷했고, 신경망 우위는 시뮬레이션에서 보고되었다.

## 연구 질문과 가설

주 질문은 하나이고, 나머지는 그 답을 검증하는 하위 질문이다.

**주 질문 (RQ)**: 같은 데이터와 같은 연산 예산에서, 물리 구조를 더 많이 넣을수록 타이어 포화 구간의 예측과 제어가 좋아지는가, 아니면 어느 지점부터 이득이 없거나 해가 되는가.

| 하위 질문 | 가설 | 검증 방법 |
| --- | --- | --- |
| RQ1 구조 | 물리 잔차와 타이어 힘 학습 모델이 순수 MLP보다 멀티스텝 롤아웃에서 안정적이다 | 롤아웃 길이별 오차 곡선 비교 |
| RQ2 구간 | 모델 간 차이는 평소 주행보다 포화 구간(슬립각이 큰 구간)에서 커진다 | 슬립각 구간별 오차 분해 |
| RQ3 일반화 | 학습에 없던 μ에서는 순수 MLP의 오차가 가장 크게 늘고, 물리 구조가 있는 모델은 덜 늘어난다 | 시뮬레이터에서 μ를 바꾼 holdout 평가 |
| RQ4 불확실성 | 앙상블 분산은 학습 분포 밖에서 커지며 실제 오차와 상관된다 | 분산과 오차의 상관, 보정 곡선 |
| RQ5 제어 | 예측 정확도 순위가 MPC 경로 추종 성능 순위와 같지 않을 수 있다 | 동일 MPC에 모델만 바꿔 폐루프 비교 |

RQ5는 1스텝이나 롤아웃 지표만으로 "더 좋은 모델"을 주장하지 않기 위한 장치다. 가설이 기각되어도 결과로 보고한다.

## 방법

방법은 네 층으로 구성된다. 팀이 직접 만드는 물리 모델 계층, 그 위의 학습 모델 4종, 시뮬레이션과 실데이터를 섞은 데이터, 그리고 네 단계 평가 프로토콜이다.

### 1. 물리 모델 계층 (직접 구현)

기존 패키지는 정답 검증용으로만 쓰고, 동역학과 타이어 모델은 팀이 PyTorch 또는 JAX로 직접 미분 가능하게 구현한다. 미분 가능해야 잔차 학습, PINN 손실, MPC 선형화에 그대로 쓸 수 있다.

| 단계 | 모델 | 추가되는 물리 | 상태 |
| --- | --- | --- | --- |
| P0 | 선형 타이어 자전거 모델 | 코너링 강성 Cα | 중간 발표에서 사용 |
| P1 | 비선형 타이어 자전거 모델 | Fiala(브러시) 또는 Pacejka 포화 | 4주차까지 |
| P2 | P1 + 종방향 힘과 마찰원 결합 | Fx·Fy 결합, 전후 하중 이동 | 4주차까지 |
| P3 | 4륜(double-track) 모델 | 좌우 하중 이동, 바퀴별 슬립 | 선택, 6주차까지 |

자전거 모델의 횡방향 운동은 다음과 같다. a, b는 무게중심에서 앞·뒤 차축까지의 거리다.

```latex
\dot U_y = \frac{F_{yf}\cos\delta + F_{yr}}{m} - r U_x, \qquad \dot r = \frac{a F_{yf}\cos\delta - b F_{yr}}{I_z}
```

```latex
\alpha_f = \arctan\frac{U_y + a r}{U_x} - \delta, \qquad \alpha_r = \arctan\frac{U_y - b r}{U_x}
```

P1의 Pacejka 횡력은 다음 형태를 쓴다(부호 규약은 구현 시 통일).

```latex
F_y = -D \sin\left(C \arctan\left(B\alpha - E\left(B\alpha - \arctan(B\alpha)\right)\right)\right), \qquad D = \mu F_z
```

구현 정확도는 같은 입력에서 [CommonRoad vehicle models](https://pypi.org/project/commonroad-vehicle-models/3.0.2/)의 출력과 비교해 검증한다. 이 패키지는 운동학 단일 트랙부터 다물체 모델까지 제공하고, 버전 2.0.0부터 Pacejka 타이어를 쓰는 단일 트랙 드리프트 모델을 포함한다.

### 2. 학습 모델 후보 (모두 같은 입력, 같은 파라미터 예산)

| 기호 | 구조 | 학습 대상 | 물리의 양 |
| --- | --- | --- | --- |
| M0 | 순수 MLP (중간 발표 모델, 128×2) | 상태 미분 전체 | 없음 |
| M1 | 물리 잔차 | 물리 모델(P1 또는 P2) 출력과 실제의 차이 | 동역학 + 타이어 |
| M2 | 타이어 힘 학습 | Fyf, Fyr만 신경망, 동역학 식은 고정 | 동역학 |
| M3 | 물리 손실(PINN식) | M0 구조 + 동역학 식 위반 페널티 | 손실에만 |
| B1 | 피팅한 물리 모델 | Pacejka 계수를 같은 학습 데이터로 최적화 | 전부 |
| B2 | GP 잔차 | P1 잔차를 희소 GP로 학습 | 동역학 + 타이어 |

공정성 규칙은 세 가지다. 같은 train/val/test 분할(주행 단위)을 쓰고, 신경망 파라미터 수를 맞추며, B1은 반드시 같은 데이터로 피팅한다. 각 모델은 시드 5개 앙상블로 학습해 평균과 분산을 함께 보고한다.

### 3. 데이터

실데이터는 풀스케일 D1(GTI)·D7(Golf)과 소형 D2(RC카 한계 기동)를 주력으로 쓰고, D5는 Deep Dynamics 비교용 보조로, D3(합성)는 μ 스윕과 정답 라벨 확보용으로 쓴다. D2에는 타이어 세트별 근사 μ가 있어 실데이터로도 노면 일반화를 검증할 수 있다. (11/17 갱신: D7 추가, D4 제외)

| 데이터 | 차량·규모 | 노면·μ | 분량·형식 | 한계 구간 | 라이선스 | 우리 용도 | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1 [Spielberg 2019 공개 자료](https://purl.stanford.edu/zb950hd3384) | 풀스케일 VW GTI (1776 kg, 장비 포함) | 마른 노면 μ 1.0 (얼음·눈 실험 폴더는 공개 자료에서 비어 있음) | 주행 11개, 약 20분, 100 Hz | 최대 1.28 g, 한계 근처 10% | CC BY-NC 3.0 | 재현, 실데이터 주평가 | 채택 |
| D7 [Stanford 2025 (Mori et al.)](https://purl.stanford.edu/hh613qz0317) | 풀스케일 VW Golf (질량 추정 약 1630 kg) | 마른 노면, Thunderhill West | CSV 22개(중복 2개 제외 20개), 약 39분, 200 Hz | 1.1~1.17 g, 비선형·포화 31% (한계 직전, 넘어선 미끄러짐은 없음) | CC BY-SA 4.0 | 풀스케일 한계 실데이터 2, GTI와 혼합·차간 일반화 | 채택 (11/17 추가) |
| D2 [Węgrzynowski et al. 2024](https://arxiv.org/pdf/2408.15610) | Xray GTXE'22, 4WD 1/8 RC카, 4.5 kg | 타이어 4세트: μ≈0.65, 0.58, 0.43, 0.45 (C·D는 테이프로 마찰 감소) | 52분, 100 Hz 재샘플, IMU·모터 전류·조향각, OptiTrack 기반 속도 정답 | 뒷축 슬립각 99분위 40° 초과 | [GitHub 저장소](https://github.com/SzalonyJohny/Learning-dynamics-models-for-velocity-estimation) 공개, 저장소 MIT (데이터 별도 라이선스 표기 없음) | 포화 구간 실평가, 실데이터 μ 일반화, probing | 채택 |
| D4 [Revs Vehicle Dynamics Database](https://purl.stanford.edu/cv083fq4881) | 1963 Corvette Grand Sport, 1965 Ferrari 250 LM, 전문 드라이버 레이스 | 건조 트랙 | 세션별 CSV·MAT, 채널 설명(channels.txt), 차량 파라미터 CSV | 레이스 주행, D2 논문 기준 큰 슬립각 기동은 적음 | [ODC-By 1.0](https://purl.stanford.edu/yf219gg2055) | 다른 차량으로의 분포 이동 테스트 | 쓰지 않음 (차간 일반화는 D1↔D7로 대체) |
| D5 [Deep Dynamics 저장소](https://github.com/linklab-uva/deep-dynamics)의 IAC 데이터 | 풀스케일 AV-21 인디 레이스카 | 건조, LVMS·Putnam Park | CSV 5개, 25 Hz, 실제 주행 약 90분 | 한계 주행 아님 (서킷 최대 0.57 g, 오벌 1.1 g는 경사 덕분) | GPL-3.0 | Deep Dynamics와 비교, 팀원 IAC 실험 | 보조 |
| D6 같은 저장소의 BayesRace 시뮬 데이터 | 시뮬레이터 차량 | 단일 | npz 2개, 정답 Pacejka 계수가 설정 파일에 기재 | pure pursuit 주행 | 저장소 확인 필요 | 정답 계수가 알려진 합성 벤치마크, D3 생성기 검증 | 보조 |
| D3 자체 합성 | 직접 만든 P2·P3 시뮬레이터 | μ 0.2~1.0 스윕 | 생성량 조절 가능 | 의도적으로 포함 | 자체 | μ 일반화, 극한 상황, μ·타이어 힘 라벨 | 채택 |
| [RACECAR](https://arxiv.org/abs/2306.03252v1) | AV-21, 27세션·6.5시간 이상 | 건조 | ROS2·nuScenes 다중 센서 | 고속 | 저장소 확인 필요 | 인지 중심이라 전처리 부담이 큼. 동역학용 일부는 D5로 대체 | 제외 |
| BETTY | 여러 자율 레이싱 차량, 13시간 이상·32 TB | 다양 | 다중 센서 | 고속 | 확인 안 함 | 규모가 2개월 프로젝트에 과함 | 제외 |
| [AutoRally](https://arxiv.org/pdf/1806.00678) | 1:5, 22 kg, 최고 90 km/h | 비포장 트랙 | ROS bag | 드리프트 가능 | 확인 안 함 | 흙 노면이라 포장도로 타이어 모델 가정과 다름 | 제외(확장 후보) |

**선택 근거**

- D1과 D2만으로 "풀스케일 고마찰"과 "소형 차량 다중 μ"라는 서로 다른 두 축의 실데이터가 확보된다.
- D2 논문의 실험 설계는 우리 모델 후보와 겹친다. 순수 신경망, 물리 잔차, 신경망 타이어, μ로 스케일한 신경망 타이어를 비교했고, 타이어 A로 학습해 C에서 평가하는 제로샷 실험을 했다. 우리 RQ3의 실데이터판 기준으로 쓸 수 있다.
- D2의 μ는 세트별 근사값이라 샘플 단위 정답이 아니다. 샘플 단위 μ와 타이어 힘 라벨은 D3에서만 얻는다.
- D2 논문 본문에는 차량이 1/8 스케일로, 기여 부분에는 1:10 스케일로 적혀 있다. 차량 사양(Xray GTXE'22)을 기준으로 1/8로 적었다.

### 3-1. 합성 데이터(D3) 생성 절차

합성 데이터는 "정답 물리"를 아는 가상 차량을 직접 몰아서 만든다. 가장 중요한 원칙은 **데이터를 만드는 시뮬레이터가 학습 모델이 가정하는 물리보다 풍부해야 한다**는 것이다. 같은 식으로 만들고 같은 식으로 맞추면 M1·M2가 정답 구조를 공짜로 얻어 비교가 무의미해진다(inverse crime).

| 구분 | 생성용 "정답" 시뮬레이터 | 학습 모델이 쓰는 물리 사전지식 |
| --- | --- | --- |
| 차체 | P3 4륜 모델, 좌우·전후 하중 이동 | P1 또는 P2 자전거 모델 |
| 타이어 | Pacejka 결합 슬립 + 타이어 이완(1차 지연) | 정상상태 Pacejka 또는 Fiala |
| 구동·조향 | 조향 각속도 제한, 구동력 지연 | 입력을 그대로 사용 |
| 센서 | 잡음·편향·필터를 거친 측정값 | 측정값만 봄 |

**1단계. 정답 차량 정의**

- 풀스케일용은 [CommonRoad 차량 파라미터](https://pypi.org/project/commonroad-vehicle-models/3.0.2/)를, 소형용은 F1TENTH 1/10 파라미터([Gym-Khana 문서](https://gym-khana.readthedocs.io/en/latest/api/env.html)에 PAC2002 타이어 포함 프리셋)를 출발점으로 쓴다.
- 두 차량 세트를 만들어 D1(풀스케일)과 D2(소형) 각각의 합성 짝으로 쓴다.

**2단계. 타이어 이완 넣기**

중간 발표에서 과거 시점 입력이 필요하다고 한 이유가 타이어 이완이다. 생성기에 다음 1차 지연을 넣어야 과거 입력의 효과를 합성 데이터에서도 검증할 수 있다. σ는 이완 길이다.

```latex
\dot F_y = \frac{U_x}{\sigma}\left(F_y^{ss}(\alpha, F_z, \mu) - F_y\right)
```

**3단계. 적분**

- 내부 적분은 RK4, 1 ms 간격으로 한다(타이어 이완과 바퀴 회전 동역학이 빠르기 때문).
- 저장은 10 ms(100 Hz)로 다운샘플한다. 중간 발표의 0.01초와 같고, D2의 100 Hz와도 같다.
- 저속에서 슬립각 식이 발산하므로 Ux가 일정 속도(예: 2 m/s) 미만인 구간은 저장하지 않는다.

**4단계. 입력 만들기 (세 종류를 섞음)**

1. **트랙 주행**: F1TENTH 트랙 중심선이나 직접 만든 트랙을 경로 추종 제어기(pure pursuit 또는 단순 MPC)로 달린다. 목표 속도는 곡률 κ에서 아래 식으로 정하고, 공격성 k를 0.6~1.1로 뽑는다. k가 1을 넘으면 의도적으로 한계를 넘는다.
2. **표준 기동**: 일정 속도 램프 조향(정상선회 특성), 스텝 조향, sine-with-dwell, 조향 처프(주파수 응답), 선회 중 제동, 선회 중 가속(파워 오버스티어).
3. **랜덤 입력**: 조향과 구동력에 대역 제한된 랜덤 신호(Ornstein-Uhlenbeck 과정)를 주고 각속도 제한을 건다. 경로 추종이 만드는 분포 편향을 메운다.

```latex
v_{target}(s) = \sqrt{\frac{k\,\mu\, g}{|\kappa(s)|}}, \qquad k \in [0.6, 1.1]
```

**5단계. 도메인 스윕**

- 에피소드마다 μ를 뽑는다. 학습용 {0.3, 0.5, 0.7, 1.0}, 평가용 보간 {0.4, 0.6, 0.85}, 외삽 {0.2}로 나눈다.
- 질량 ±10%, 타이어 강성 ±15% 범위로 함께 흔든다.
- 일부 에피소드는 주행 중 μ가 바뀌는 구간(마른 노면에서 빙판 구간)을 넣어 적응 실험에 쓴다.
- 차체 슬립각이 일정 값(예: 30°)을 넘거나 스핀하면 에피소드를 끝낸다. 드리프트 구간을 원하면 이 기준을 완화한 별도 세트로 만든다.

**6단계. 센서 모델**

- r, a_y, Ux, Uy에 가우시안 잡음과 작은 편향을 넣는다. 실제 Uy는 GPS/INS나 모션캡처 추정값이므로 잡음을 가장 크게 둔다.
- 학습 타깃 ṙ, U̇y는 실데이터와 똑같이 잡음 있는 측정값의 차분이나 필터로 만든다. 참 미분값은 진단용으로만 따로 저장한다.
- 실데이터 전처리 코드와 같은 코드를 지나가게 해서 "합성용 파이프라인"이 따로 생기지 않게 한다.

**7단계. 저장과 라벨**

에피소드마다 하나의 파일로 저장하고, 메타데이터에 다음을 기록한다.

- 시계열: 상태, 입력, 측정값
- 정답 라벨: 바퀴별 타이어 힘, 슬립각, μ, 차량 파라미터
- 구간 라벨: |α|/α_peak로 계산한 평소 주행, 비선형 진입, 포화 구분
- 생성 조건: 입력 종류, k, 시드

분할은 반드시 에피소드 단위로 하고, μ holdout과 트랙 holdout을 따로 둔다.

**8단계. 생성기 검증 (학습 전에 반드시)**

1. 같은 입력에서 우리 P2와 CommonRoad 단일 트랙 드리프트 모델의 출력을 비교한다.
2. 모든 샘플에서 마찰원 조건 √(Fx² + Fy²) ≤ μFz가 지켜지는지 확인한다.
3. 정상선회 램프 조향 결과로 언더스티어 그래디언트와 최대 횡가속도(약 μg)가 이론값과 맞는지 본다.
4. D1·D2와 슬립각·횡가속도 히스토그램을 겹쳐 보고, 합성 분포가 실데이터 범위를 덮는지 확인한다.
5. D6(정답 Pacejka 계수가 알려진 BayesRace 데이터)에 B1 피팅을 돌려 계수를 복원하는지 보고, 피팅 코드 자체를 검증한다.

**규모 감**: 에피소드 2,000개 × 20초 × 100 Hz면 약 400만 샘플이다. 벡터화하면 CPU로도 충분할 것으로 보지만, 1주차에 실제 생성 시간을 측정해 확정한다.

**한계**: D3에서의 결론은 "우리가 만든 물리 안에서"의 결론이다. 보고서에서는 D3 결과와 D1·D2 결과를 항상 나란히 두고, 실데이터에서 같은 경향이 나올 때만 일반적인 주장으로 쓴다.

### 4. 평가 프로토콜

1. **1스텝 예측**: ṙ와 U̇y를 각각 표준화한 MSE를 따로 보고한다. 합산 시 가중치를 명시한다.
2. **멀티스텝 롤아웃**: 0.01초 간격으로 0.1, 0.5, 1.0초까지 자기 예측을 다시 입력해 누적 오차를 본다. 중간 발표의 "0.05초 예측"을 확장한 것이다.
3. **구간별 오차**: 뒷바퀴 슬립각 또는 |a_y|/(μg)로 평소 주행, 비선형 진입, 포화 구간을 나눠 보고한다.
4. **폐루프 제어**: 같은 NMPC에 동역학 모델만 바꿔 넣고 D3 시뮬레이터 트랙에서 경로 추종 오차(횡방향 오차 RMS, 최대값)와 이탈 여부를 비교한다.

불확실성은 앙상블 분산과 실제 오차의 상관, 그리고 학습에 없던 μ에서 분산이 커지는지로 평가한다.

**결정 사항 (11/7)**
- 주 지표는 0.5초·1.0초 롤아웃 오차다. 1스텝 오차는 원논문 대조용 보조 지표로 쓴다.
- 주 실험에서는 모델에 노면 μ를 주지 않는다. 보조 실험으로 μ를 준 조건을 같이 보고한다.

## 도구 스택

시뮬레이터는 직접 만든 Python 코드를 중심으로 하고, 외부 도구는 검증·트랙·제어·시각화에만 쓴다. 대형 3D 시뮬레이터는 선택 사항으로 둔다.

| 영역 | 도구 | 쓰는 이유 | 비고 |
| --- | --- | --- | --- |
| 동역학 구현 | PyTorch 또는 JAX | 미분 가능한 물리 모델, 학습과 같은 코드 | 팀 내 하나로 통일 |
| 물리 검증 | [CommonRoad vehicle models](https://pypi.org/project/commonroad-vehicle-models/3.0.2/) | `pip install`로 설치, 운동학부터 다물체까지 기준 구현 | 우리 구현의 정답 비교용 |
| 트랙·RL 환경 | [F1TENTH gym](https://arxiv.org/pdf/2402.18558) 계열, [Gym-Khana](https://gym-khana.readthedocs.io/en/stable/) | F1TENTH gym은 CommonRoad 단일 트랙 모델 기반. Gym-Khana는 이를 확장해 PAC2002 타이어 드리프트 모델, 다물체 모델, 디버그 오버레이 렌더링 제공 | 실제 트랙 레이아웃 재사용, 1/10 스케일 파라미터 |
| 최적 제어 | CasADi + [acados](https://github.com/acados/acados) | acados는 NMPC용 고속 솔버로 Python 인터페이스와 CasADi 기호 표현을 지원 | 처음엔 CasADi + IPOPT로 시작 |
| 학습 모델 in MPC | [L4acados](https://arxiv.org/html/2411.19258v2) | Python 기반 잔차 모델(GP, 신경망)을 acados MPC에 넣는 프레임워크 | M1·B2를 MPC에 넣을 때 |
| 시각화 | [Rerun](https://github.com/rerun-io/rerun) | `pip install rerun-sdk`, 시계열·2D·3D를 시간축으로 기록하고 되돌려 보기 | 롤아웃 비교, 주행 리플레이 |
| 실험 관리 | Weights & Biases | 모델 × 시드 × 데이터 매트릭스 추적 | 무료 학생 플랜 여부 확인 |
| 정적 그림 | matplotlib | 보고서용 오차 곡선, 구간별 막대 | 그림 스타일 통일 |
| 3D 시뮬레이터 (선택) | [BeamNG.tech](https://documentation.beamng.com/beamng_tech/) + BeamNGpy | 소프트바디 물리 기반 연구용 버전, Python API로 차량 제어와 텔레메트리 수집 | 연구용 빌드는 신청 필요, Windows 위주. 시간이 남으면 "우리 모델이 모르는 물리" 테스트베드 |

**선택 기준**: 2개월 안에 5명이 같이 쓰려면 설치가 pip 수준이어야 하고, GPU 없이도 시뮬레이션이 돌아야 한다. 그래서 CARLA, BeamNG 같은 대형 시뮬레이터를 핵심 경로에서 뺐다.

**시각화 산출물 계획**

- 트랙 위 궤적 겹쳐 그리기: 실제 vs 모델별 롤아웃
- 슬립각–횡력 산점도 위에 각 모델이 학습한 타이어 곡선 겹치기 (중간 발표 개념도의 실데이터판)
- 롤아웃 길이별 오차 곡선, 슬립각 구간별 오차 막대
- μ 스윕에서 오차와 앙상블 분산을 함께 보는 그림
- 발표용 주행 리플레이 영상 (Rerun 또는 Gym-Khana 렌더링)

## 8주 일정과 마일스톤

1~2주차에 데이터와 인터페이스를 고정하고, 4주차 말 중간 점검에서 실험 범위를 확정한 뒤, 후반 3주를 일반화·제어 실험과 정리에 쓴다. 일정은 2026년 11월 9일 시작, 2027년 1월 4일 최종 발표로 가정했다(발표일은 확정 필요).

[일정 도식: 작업 9개, 마일스톤 2개 — 원본 문서의 그림이며 .md에서는 아래 표로 대체]

| 작업 | 기간 |
| --- | --- |
| 데이터 확인·API 고정 | 11/9~11/22 |
| 물리 모델 P1·P2 | 11/9~11/29 |
| 합성 데이터 D3 | 11/23~12/6 |
| 학습 모델 M0~M3, B1·B2 | 11/23~12/13 |
| 4륜 모델 P3 (선택) | 11/30~12/13 |
| 롤아웃·구간별 평가 | 11/30~12/20 |
| **중간 점검** | **12/6** |
| μ 일반화·불확실성 | 12/7~12/20 |
| NMPC 폐루프 실험 | 12/7~12/27 |
| 보고서·발표 준비 | 12/21~1/3 |
| **최종 발표** | **1/4** |

**12월 6일 중간 점검 통과 기준**: P1·P2가 CommonRoad와 일치하고, D1 기준으로 M0~M3의 1스텝·롤아웃 결과가 나와 있어야 한다. 미달이면 P3와 MPC를 축소하고 롤아웃 평가에 집중한다.

| 주차 | 기간 | 끝날 때 나와 있어야 할 것 |
| --- | --- | --- |
| 1 | 11/9~11/15 | D2 사용 가능 여부, 원논문 대조표, 저장소 구조 |
| 2 | 11/16~11/22 | 데이터 로더·모델 API 고정, P1 구현 |
| 3 | 11/23~11/29 | P2 검증 완료, D3 생성기 초안, M0 재학습(새 평가 코드) |
| 4 | 11/30~12/6 | M1~M3 1차 결과, 구간별 오차 첫 그림 → 중간 점검 |
| 5 | 12/7~12/13 | 전 모델 앙상블 학습, P3(선택), 단순 NMPC 동작 |
| 6 | 12/14~12/20 | μ holdout·불확실성 분석, probing |
| 7 | 12/21~12/27 | 폐루프 비교 완료, 그림 확정 |
| 8 | 12/28~1/3 | 보고서·발표 자료·데모 영상 |

## 5인 역할 분담

역할은 모델 하나씩이 아니라 파이프라인 단계별로 나눈다. 모든 모델이 같은 데이터, 같은 평가 코드를 지나가야 비교가 공정하기 때문이다. 이름은 팀이 정해 채운다.

| 역할 | 담당 | 주요 산출물 | 협업 지점 |
| --- | --- | --- | --- |
| A 물리 모델 | (미정) | P0~P3 구현, CommonRoad 대조 테스트, Pacejka 피팅(B1) | B, C에 미분 가능한 모델 제공 |
| B 데이터·시뮬레이터 | (미정) | D1·D2 전처리와 주행 단위 분할, D3 합성 데이터 생성기, μ 스윕 | 모든 역할이 쓰는 데이터 로더 |
| C 학습 모델 | (미정) | M0~M3, B2 학습 코드, 앙상블, 파라미터 수 맞추기 | A의 물리 모델을 모듈로 호출 |
| D 평가·분석 | (미정) | 4단계 평가 프로토콜 코드, 구간별 오차, 불확실성 분석, probing | C의 모델을 체크포인트로 받아 일괄 평가 |
| E 제어·시각화 (PM 겸임) | (미정) | NMPC 폐루프 실험, Rerun 시각화, 일정·저장소 관리, 발표 자료 통합 | D와 함께 최종 그림 확정 |

**운영 규칙**

- 주 1회 전체 회의에서 실험 매트릭스(모델 × 데이터 × 지표) 표를 갱신한다.
- 저장소는 `physics/`, `data/`, `models/`, `eval/`, `control/`, `viz/`로 역할과 같게 나눈다.
- 평가 코드는 D만 수정하고, 다른 역할은 결과를 같은 스크립트로만 낸다.
- 각자 맡은 영역의 물리나 수식을 주간 회의에서 10분씩 설명한다. 학회 프로젝트로서 팀 전원이 차량 동역학을 이해하는 것이 목표다.

## 자원·예산, 위험과 대응

연산 부담이 작은 프로젝트이므로 GPU 예산 약 50만 원은 여유가 있다. 중간 발표 기준 MLP 1회 학습은 RTX 4070 Ti에서 약 18분이었다.

**연산량 추정** (가정: 모델 6종 × 시드 5개 × 데이터 3종 ≈ 90회 학습)

- 1회 20분 기준 약 30 GPU시간이다. 하이퍼파라미터 탐색을 3배로 잡아도 100 GPU시간 안쪽이다.
- 시뮬레이션 데이터 생성과 MPC는 CPU 작업이다.
- 남는 예산은 앙상블 확대와 롤아웃 길이 확장에 쓴다. 클라우드 시간당 단가는 업체별로 확인이 필요하다.

| 위험 | 가능성 | 영향 | 대응 |
| --- | --- | --- | --- |
| D2 데이터가 형식·스케일·라이선스 문제로 사용 불가 | 중 | 중 | 1주차에 먼저 확인한다. 불가하면 D1 + D3만으로 진행하고 실데이터 한계 구간 평가는 한계로 명시한다 |
| MPC 통합이 일정 안에 안 됨 | 중 | 중 | CasADi + IPOPT 단순 NMPC로 낮춘다. 그래도 안 되면 RQ5를 롤아웃 기반 오픈루프 평가로 대체한다 |
| 신경망 모델이 MPC에서 너무 느림 | 중 | 저 | 작은 은닉층, 선형화 MPC, 또는 롤아웃 평가로 범위를 줄인다 |
| 4륜 모델(P3) 구현 지연 | 중 | 저 | P3는 선택 사항이다. P2까지로 결론을 낸다 |
| 합성 데이터만으로 결론이 나는 순환 논리 | 고 | 고 | D3 결론은 "우리 물리 하에서"로 한정하고, 실데이터 D1·D2 결과와 구분해 보고한다 |
| 물리 baseline이 약하게 설정되어 격차가 부풀려짐 | 중 | 고 | B1은 같은 데이터로 피팅하고 피팅 절차를 공개한다 |
| 역할 간 인터페이스 불일치 | 중 | 중 | 2주차까지 데이터 로더와 모델 API를 먼저 고정한다 |

## 최종 산출물과 확인 필요 사항

**최종 산출물**

- 최종 발표 자료와 보고서(학회 논문 형식, 4~6쪽 분량 목표)
- 재현 가능한 코드 저장소: 물리 모델, 데이터 생성기, 학습·평가 스크립트, 설정 파일
- 결과 표: 모델 × 데이터 × 지표 매트릭스
- 시각화 데모: 트랙 주행 리플레이와 롤아웃 비교

**바로 확인할 것**

- [ ] Spielberg 2019 본문·Supplementary에서 입력 구성, 은닉층, 사전학습 순서, 손실 정의 대조
- [ ] D2 데이터 다운로드 (GitHub 저장소, ROS2 bag + 전처리 CSV) 및 데이터 라이선스 저자 문의
- [ ] Spielberg 공개 데이터의 주행 수와 길이 (분할 설계용)
- [ ] 최종 발표일 확정 (일정표는 2027년 1월 첫 주 종료로 가정)
- [ ] 역할별 담당자 배정

## 참고문헌

- [Spielberg et al. 2019, 데이터·코드 저장소](https://purl.stanford.edu/zb950hd3384)
- [Spielberg, Leveraging learning for vehicle control at the limits of handling (박사논문)](https://purl.stanford.edu/wp128xy5795)
- [Stanford 보도, 2019](https://iconnect007.com/article/116402/stanford-autonomous-car-learns-to-handle-unknown-conditions/116405/ein)
- [Broadbent et al., Neural Network Tire Force Modeling for Automated Drifting (AVEC '24)](https://arxiv.org/abs/2407.13760)
- [Neural-ExpTanh / NODE tire models](https://arxiv.org/html/2306.06330v2)
- [First, Learn What You Don't Know](https://arxiv.org/pdf/2411.00107)
- [TRI, Drifting with Unknown Tires](https://www.tri.global/research/drifting-unknown-tires-learning-vehicle-models-online-neural-networks-and-model-predictive)
- [PINN for Vehicle Lateral Dynamics (Chalmers)](https://odr.chalmers.se/bitstreams/ab7f406a-79f1-43a9-8d35-0bc5de66de0c/download)
- [Fine-Tuning Hybrid PINN (FTHD)](https://www.arxiv.org/pdf/2409.19647)
- [BayesRace](https://arxiv.org/pdf/2005.04755)
- [Ning & Behl, Vehicle Dynamics Modeling Using GPs](https://arxiv.org/abs/2306.03405v1)
- [Węgrzynowski et al., Learning dynamics models for velocity estimation](https://arxiv.org/abs/2408.15610v1)
- [Węgrzynowski et al. 데이터·코드 저장소 (GitHub)](https://github.com/SzalonyJohny/Learning-dynamics-models-for-velocity-estimation)
- [CommonRoad vehicle models (PyPI)](https://pypi.org/project/commonroad-vehicle-models/3.0.2/)
- [Unifying F1TENTH Autonomous Racing (F1TENTH gym 모델 설명)](https://arxiv.org/pdf/2402.18558)
- [Gym-Khana 문서](https://gym-khana.readthedocs.io/en/stable/)
- [acados](https://github.com/acados/acados), [L4acados](https://arxiv.org/html/2411.19258v2)
- [Rerun](https://github.com/rerun-io/rerun)
- [BeamNG.tech 문서](https://documentation.beamng.com/beamng_tech/)
- [Revs Vehicle Dynamics Database (Stanford)](https://purl.stanford.edu/cv083fq4881), [2013 Targa Sixty-Six](https://purl.stanford.edu/yf219gg2055), [2014 Targa Sixty-Six](https://purl.stanford.edu/hd122pw0365)
- [Deep Dynamics 저장소 (IAC·BayesRace 데이터)](https://github.imc.re/linklab-uva/deep-dynamics), [Deep Dynamics 논문](https://arxiv.org/abs/2312.04374v2)
- [RACECAR dataset](https://arxiv.org/abs/2306.03252v1)
- [AutoRally](https://arxiv.org/pdf/1806.00678)
