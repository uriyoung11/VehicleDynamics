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


def load(name, split, **kw):
    """name: "d1", "d2", "d2-tire"(μ 일반화 분할), "d3:<생성 데이터 폴더>"."""
    if name == "d1":
        return load_d1(split)
    if name == "d2":
        return load_d2(split, mode="run", **kw)
    if name == "d2-tire":
        return load_d2(split, mode="tire", **kw)
    if name.startswith("d3:"):
        return load_d3(name[3:], split, **kw)
    raise ValueError(name)


def windows(episodes, T=4):
    """1스텝 학습용: 입력 (n, T, 5) [r, Uy, Ux, delta, long/long_scale], 타깃 (n, 2) 다음 시점 [r, Uy],
    그리고 타깃 시점의 포화 정도 (n,)."""
    Xs, Ys, Rs = [], [], []
    for ep in episodes:
        F = ep["F"].copy(); F[:, 4] /= ep["long_scale"]
        n = len(F) - T
        if n <= 0:
            continue
        idx = np.arange(n)[:, None] + np.arange(T)[None, :]
        Xs.append(F[idx]); Ys.append(F[T:, :2]); Rs.append(ep["regime"][T:])
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
