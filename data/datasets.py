"""D1, D2, D3를 같은 형식의 에피소드 목록으로 읽는다.

에피소드 = 한 번의 연속 주행. dict 형식:
  name, dataset, split, mu, dt, vehicle(physics 프리셋 이름)
  F       (N, 5) float64: [r, Uy, Ux, delta, long]   ← Spielberg 입력 순서
  long_scale  : 모델 입력에 넣을 때 long을 나눌 값
  regime  (N,)  : max(|α_f|/α_peak_f, |α_r|/α_peak_r). <0.5 평소, 0.5~1 비선형 진입, >1 포화
  long_kind   : "force_front" [N] (D1, D3) 또는 "motor_current" [A] (D2)

분할(split)은 항상 주행 단위다. 같은 주행이 train과 test에 동시에 들어가지 않는다.

  from data.datasets import load
  train = load("d1", "train"); test = load("d2", "test")
  d3 = load("d3:data/generated/f1tenth_v3_trial", "test_mu_interp")
"""
import glob, os
import numpy as np
import pandas as pd

from physics import PRESETS
from physics.tires import peak_slip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D1_DIR = os.path.join(ROOT, "data", "scirob_submission", "Model_Learning", "data", "exp", "exp_data", "dry")
D2_DIR = os.path.join(ROOT, "data", "d2")

# D1: 원본 주행 파일 기준 주행 단위 분할 (repro/spielberg2019와 같음)
D1_SPLIT = {"val": ["exp_11", "exp_12", "exp_15"], "test": ["exp_5", "exp_9"]}
D1_TRIM = 20  # 각 주행 앞 0.2 s 필터 시작 구간 제거

# D2: 파일 → 타이어 세트, μ (GitHub 저장소 dataset/data/tire_* 폴더 이름으로 대응)
D2_FILES = {
    "hoons_all.csv": ("A", 0.65),
    "iros_xray_30_7.csv": ("B", 0.58),
    "iros_hoons_malarska_23_5.csv": ("C", 0.43),
    "xray_duck_test.csv": ("D", 0.45),
}
D2_MIN_RUN_S = 1.0          # 이보다 짧은 주행은 버림
D2_TIRE_SPLIT = {"train": ["A", "B"], "test": ["C", "D"]}  # μ 일반화(제로샷) 분할


def regime_ratio(F, vehicle, mu):
    """측정 상태로 슬립각을 계산해 축별 포화 정도(|α|/α_peak)를 낸다. 반환: (앞, 뒤)."""
    p = PRESETS[vehicle]
    r, uy, ux, delta = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
    af = np.arctan((uy + p.a * r) / ux) - delta
    ar = np.arctan((uy - p.b * r) / ux)
    return np.abs(af) / peak_slip(p.tire_f, mu), np.abs(ar) / peak_slip(p.tire_r, mu)


def _episode(name, dataset, split, mu, F, vehicle, long_scale, long_kind, regime=None, regime_rear=None):
    """regime: 앞·뒤 중 큰 쪽(기본 구간 라벨), regime_rear: 뒷바퀴만(오버스티어 쪽)."""
    F = np.asarray(F, dtype=np.float64)
    rf, rr = regime_ratio(F, vehicle, mu)
    return dict(name=name, dataset=dataset, split=split, mu=float(mu), dt=0.01, vehicle=vehicle, F=F,
                long_scale=long_scale, long_kind=long_kind,
                regime=np.maximum(rf, rr) if regime is None else np.asarray(regime, np.float64),
                regime_rear=rr if regime_rear is None else np.asarray(regime_rear, np.float64))


def load_d1(split):
    out = []
    for f in sorted(glob.glob(os.path.join(D1_DIR, "exp_*.npz")), key=lambda s: int(s.split("_")[-1][:-4])):
        name = os.path.basename(f)[:-4]
        s = next((k for k, runs in D1_SPLIT.items() if name in runs), "train")
        if split not in ("all", s):
            continue
        z = np.load(f)
        F = np.stack([z["r"], z["uy"], z["ux"], z["delta"], z["fxf"]], 1)[D1_TRIM:]
        out.append(_episode(f"d1/{name}", "d1", s, 1.0, F, "fullscale", PRESETS["fullscale"].m * 100, "force_front"))
    return out


def _d2_run_splits(runs, seed, frac=(0.7, 0.15, 0.15)):
    """주행 목록을 샘플 수 기준 약 70/15/15로 나눈다(타이어 세트마다 따로)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(runs))
    total = sum(n for _, n in runs)
    acc, out = 0, {}
    for i in order:
        rid, n = runs[i]
        out[rid] = "train" if acc < frac[0] * total else "val" if acc < (frac[0] + frac[1]) * total else "test"
        acc += n
    return out


def load_d2(split, mode="run", seed=0):
    """mode="run": 타이어 세트마다 주행 단위 70/15/15.
    mode="tire": 타이어 A, B로 학습(그중 15%는 val), C, D로 시험 (μ 일반화)."""
    out = []
    for fname, (tire, mu) in D2_FILES.items():
        d = pd.read_csv(os.path.join(D2_DIR, fname), index_col=0)
        groups = [(rid, g) for rid, g in d.groupby("run_id") if len(g) >= D2_MIN_RUN_S * 100]
        sp = _d2_run_splits([(rid, len(g)) for rid, g in groups], seed)
        for rid, g in groups:
            s = sp[rid]
            if mode == "tire":
                s = "test" if tire in D2_TIRE_SPLIT["test"] else ("val" if s == "val" else "train")
            if split not in ("all", s):
                continue
            F = g[["r", "v_y", "v_x", "delta", "Iq"]].to_numpy()
            out.append(_episode(f"d2/{tire}/{fname[:-4]}/run{rid}", "d2", s, mu, F, "f1tenth", 1000.0, "motor_current"))
    return out


def load_d3(root, split, use_meas=True):
    from data.loader import load_manifest, load_episode, load_config
    cfg = load_config(root)
    vehicle = next(k for k, v in PRESETS.items() if v.name == cfg["vehicle"]["name"])
    out = []
    for row in load_manifest(root):
        if split not in ("all", row["split"]) or row["n_samples"] < 10:
            continue
        ep = load_episode(root, row)
        g = (lambda c: ep["meas_" + c]) if use_meas and "meas_r" in ep else (lambda c: ep[c])
        F = np.stack([g("r"), g("uy"), g("ux"), ep["delta"], ep["fxf"]], 1)
        # D3는 참값 슬립각으로 라벨을 붙인다(질량·강성 흔들기 전 공칭 α_peak 기준)
        rear = np.abs(ep["alpha_r"]) / peak_slip(PRESETS[vehicle].tire_r, row["mu"])
        out.append(_episode(f"d3/{os.path.basename(root)}/ep{row['episode']}", "d3", row["split"], row["mu"], F, vehicle,
                            PRESETS[vehicle].m * 100, "force_front", regime=ep["regime_ratio"], regime_rear=rear))
    return out


# D7: Stanford 2025 VW Golf, Thunderhill West (Mori et al. 2025, purl.stanford.edu/hh613qz0317, CC BY-SA 4.0)
D7_DIR = os.path.join(ROOT, "data", "d7")
D7_DUP = {"2023-10_2", "2023-10_3"}            # 2023-10, 2023-10_1과 바이트 단위로 같은 파일(md5 확인)
# [팀 결정 11/19] 팀원 분할로 통일 (data/d7_split_team.json = 팀원 저장소 deep_dynamics/data/golf/split.json, 시드 0).
# 이전 우리 분할: val 10_6·11_1·11_10, test 10_4·11_3·11_8 (11/19 이전 사다리 결과 ①~⑥·차량 간 평가는 이 분할)
D7_SPLIT = {"val": ["2023-11_4", "2023-11_7", "2023-11_8"], "test": ["2023-10_1", "2023-11_1", "2023-11_5"]}
D7_MIN_UX = 2.0                                # D1·풀스케일 프리셋의 ux_min_valid와 같음


# D7 액추에이터 기반 종방향 힘 보정값 (data/d7 8개 파일, Ux > 5 m/s, 최소제곱, R² 0.937):
#   가속도 ≈ K_DRIVE·(엔진 토크 × GearRatio) − K_BRAKE·(브레이크 힘 합) + ...  → 질량 ≈ 1/K_BRAKE ≈ 1630 kg
D7_K_DRIVE, D7_K_BRAKE = 6.5826e-4, 6.1283e-4


def load_d7(split, long="accel"):
    """long: "accel" 측정 가속도로 만든 앞바퀴 힘(기본. [r, Uy]만 예측할 때),
             "actuator" 엔진 토크·기어비·브레이크 힘으로 만든 앞바퀴 힘(미래 정보 없음. Ux까지 예측할 때),
             "pedal" 스로틀/100(제동 중이면 −브레이크 압력/최대). Deep Dynamics·팀원 형식."""
    return _load_d7(split, long)


def _load_d7(split, long):
    """200 Hz → 100 Hz(두 샘플 평균). Ux > 2 m/s인 연속 구간을 에피소드로.
    종방향 입력은 D1의 fxf(앞바퀴 축 종방향 힘)와 같은 뜻으로 만든다:
      전체 종방향 힘/m = dUx/dt − Uy·r (Ux를 0.09 s 평활 후 미분),
      앞바퀴 몫 = 가속 1 (전륜구동), 제동 (앞 브레이크 힘)/(전체 브레이크 힘).
    모델 입력은 long/long_scale(= m·100)이라 질량이 약분된다. Golf 질량을 몰라도 된다.
    주의: D1의 fxf는 제어 명령/추정값, D7은 측정 가속도에서 계산한 값이라 성격이 조금 다르다.
    차량 프리셋은 GTI(fullscale)를 쓴다(Golf 제원 미확인). 포화 라벨은 μ=1.0 가정."""
    from scipy.ndimage import uniform_filter1d
    m = PRESETS["fullscale"].m
    out = []
    for f in sorted(glob.glob(os.path.join(D7_DIR, "*.csv"))):
        name = os.path.basename(f)[:-4]
        if name in D7_DUP:
            continue
        s = next((k for k, runs in D7_SPLIT.items() if name in runs), "train")
        if split not in ("all", s):
            continue
        d = pd.read_csv(f, comment="#")
        cols = ["yawRate_radps", "Vy_mps", "Vx_mps", "roadWheelAngle_rad", "brake_fl_N", "brake_fr_N", "brake_rl_N", "brake_rr_N",
                "engineTorque_Nm", "GearRatio", "throttleCmd_percent", "brake_fl_bar"]
        a = d[cols].to_numpy(np.float64)
        n2 = len(a) // 2 * 2
        a = a[:n2].reshape(-1, 2, len(cols)).mean(1)               # 100 Hz
        r, uy, ux, delta = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bf, bt = a[:, 4] + a[:, 5], a[:, 4:8].sum(1)
        if long == "accel":
            ax = np.gradient(uniform_filter1d(ux, 9), 0.01) - uy * r
            share = np.where((ax < 0) & (bt > 1.0), bf / np.maximum(bt, 1e-9), 1.0)
            fxf, ls, kind = share * m * ax, m * 100, "force_front"
        elif long == "actuator":
            # 앞바퀴 = 구동(전륜) − 앞 브레이크. 단위를 D1과 맞추려고 GTI 질량 기준 가속도로 환산
            drive = D7_K_DRIVE / D7_K_BRAKE * a[:, 8] * a[:, 9]          # [N] (Golf 추정 질량 기준)
            fxf, ls, kind = (drive - bf) * (m * D7_K_BRAKE), m * 100, "force_front"
        else:
            fxf = np.where(a[:, 11] > 0.5, -a[:, 11] / 50.0, a[:, 10] / 100.0)
            ls, kind = 1.0, "pedal"
        ok = ux > D7_MIN_UX
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for k, (i0, i1) in enumerate(zip(edges[::2], edges[1::2])):
            if i1 - i0 < 100:
                continue
            F = np.stack([r, uy, ux, delta, fxf], 1)[i0:i1]
            out.append(_episode(f"d7/{name}/seg{k}", "d7", s, 1.0, F, "fullscale", ls, kind))
    return out


def decimate(eps, k):
    """시간 간격을 k배로: k 샘플 평균(포화 라벨은 최댓값). dt도 k배."""
    out = []
    for e in eps:
        n = len(e["F"]) // k * k
        if n < 2 * k:
            continue
        g = lambda a, f: f(np.asarray(a)[:n].reshape(n // k, k, *np.asarray(a).shape[1:]), axis=1)
        out.append(dict(e, F=g(e["F"], np.mean), regime=g(e["regime"], np.max), regime_rear=g(e["regime_rear"], np.max),
                        dt=e["dt"] * k))
    return out


VEH_ID = {"d1": 0.0, "d7": 1.0}


def _with_vehicle_flag(eps):
    """6번째 입력 채널로 차 구분 값(D1 GTI 0, D7 Golf 1)을 붙인다. 물리 부분은 0~4번 채널만 쓰므로 영향 없다."""
    for e in eps:
        e["F"] = np.c_[e["F"], np.full(len(e["F"]), VEH_ID[e["dataset"]])]
    return eps


def load(name, split, **kw):
    """name: "d1", "d2", "d2-tire"(μ 일반화 분할), "d3:<생성 데이터 폴더>", "d7".
    여러 데이터 합치기: "d1+d7". 끝에 "|veh"를 붙이면 차 구분 채널을 추가한다 (예: "d1+d7|veh", 평가용 "d1|veh")."""
    if "@" in name and not name.startswith("d3:"):          # "d1+d7@2": 시간 간격 2배(0.02 s)
        base, k = name.rsplit("@", 1)
        return decimate(load(base, split, **kw), int(k))
    if name.endswith("|veh"):
        return _with_vehicle_flag(load(name[:-4], split, **kw))
    if "+" in name and not name.startswith("d3:"):
        return [e for part in name.split("+") for e in load(part, split, **kw)]
    if name in ("d7", "d7a", "d7p"):
        return [dict(e, dataset="d7") for e in load_d7(split, {"d7": "accel", "d7a": "actuator", "d7p": "pedal"}[name])]
    if name == "d1":
        return load_d1(split)
    if name == "d2":
        return load_d2(split, mode="run", **kw)
    if name == "d2-tire":
        return load_d2(split, mode="tire", **kw)
    if name.startswith("d3:"):
        return load_d3(name[3:], split, **kw)
    raise ValueError(name)


def windows(episodes, T=4, n_out=2):
    """1스텝 학습용: 입력 (n, T, 5) [r, Uy, Ux, delta, long/long_scale], 타깃 (n, n_out) 다음 시점 [r, Uy(, Ux)],
    그리고 타깃 시점의 포화 정도 (n,)."""
    Xs, Ys, Rs = [], [], []
    for ep in episodes:
        F = ep["F"].copy(); F[:, 4] /= ep["long_scale"]   # 5번째 이후 채널(차 구분 등)은 그대로
        n = len(F) - T
        if n <= 0:
            continue
        idx = np.arange(n)[:, None] + np.arange(T)[None, :]
        Xs.append(F[idx]); Ys.append(F[T:, :n_out]); Rs.append(ep["regime"][T:])
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Rs)


def summary(episodes):
    n = sum(len(e["F"]) for e in episodes)
    reg = np.concatenate([e["regime"] for e in episodes]) if episodes else np.zeros(0)
    rear = np.concatenate([e["regime_rear"] for e in episodes]) if episodes else np.zeros(0)
    return dict(episodes=len(episodes), minutes=round(n / 6000, 1),
                mu=sorted({e["mu"] for e in episodes}),
                regime_pct={k: round(100 * float(v), 1) for k, v in
                            dict(normal=np.mean(reg < 0.5), nonlinear=np.mean((reg >= 0.5) & (reg < 1)), saturated=np.mean(reg >= 1)).items()},
                rear_saturated_pct=round(100 * float(np.mean(rear >= 1)), 1))
