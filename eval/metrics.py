"""평가 지표. 모든 모델은 models.baselines의 인터페이스(hist -> 다음 [r, Uy])를 따른다.

1스텝 nMSE: (예측 미분 - 실제 미분)^2 / 실제 미분 분산. r, Uy 따로. 0이면 완벽, 1이면 "평균만 맞히는" 수준.
롤아웃 nRMSE: 처음 T시점은 실제 값으로 시작하고, 이후로는 r, Uy에 모델 자신의 예측을 다시 넣는다.
             Ux, 조향, 종방향 입력은 실제 값을 쓴다(모델이 Ux를 예측하지 않기 때문).
             h스텝 뒤 오차의 RMSE를 평가 데이터의 r, Uy 표준편차로 나눈다.
구간: 1스텝은 타깃 시점, 롤아웃은 구간 안 최댓값의 포화 정도로 나눈다.
  normal < 0.5 <= nonlinear < 1.0 <= saturated
"""
import numpy as np
import torch

from data.datasets import windows

BINS = {"normal": (0.0, 0.5), "nonlinear": (0.5, 1.0), "saturated": (1.0, np.inf)}
HORIZONS = {"0.1s": 10, "0.5s": 50, "1.0s": 100}       # 0.01 s 간격 기준 스텝 수 (기존 결과와 호환)
HORIZON_S = {"0.1s": 0.1, "0.5s": 0.5, "1.0s": 1.0}   # 간격이 다른 데이터는 초 단위로 환산한다


def _bins(reg):
    out = {"all": np.ones_like(reg, bool)}
    for k, (lo, hi) in BINS.items():
        out[k] = (reg >= lo) & (reg < hi)
    return out


@torch.no_grad()
def one_step(model, episodes, T=4, batch=50000, rear=False):
    n = getattr(model, "n_out", 2)
    X, Y, R = windows(episodes, T, n)
    if rear:
        R = np.concatenate([e["regime_rear"][T:] for e in episodes if len(e["F"]) > T])
    dt = episodes[0]["dt"]
    Xt = torch.as_tensor(X, dtype=torch.float64)
    P = torch.cat([model(Xt[i:i + batch]) for i in range(0, len(Xt), batch)]).numpy()
    d_true, d_pred = (Y - X[:, -1, :n]) / dt, (P - X[:, -1, :n]) / dt
    var = d_true.var(0)
    se = (d_pred - d_true) ** 2
    res = {}
    for k, m in _bins(R).items():
        if m.sum() == 0:
            continue
        v = se[m].mean(0) / var
        # mean은 항상 r, Uy 두 채널 평균(출력 형식이 달라도 비교 가능하게). Ux는 따로
        res[k] = dict(n=int(m.sum()), r=float(v[0]), uy=float(v[1]), mean=float(v[:2].mean()),
                      state_mse=float(((P[m][:, :2] - Y[m][:, :2]) ** 2).mean()), **({"ux": float(v[2])} if n == 3 else {}))
    return res


@torch.no_grad()
def rollout(model, episodes, T=4, horizons=None, stride=25, rear=False):
    """horizons=None이면 HORIZON_S(초)를 데이터 간격으로 환산. stride는 0.01 s 기준 스텝 수(간격이 다르면 같은 시간으로 환산)."""
    n = getattr(model, "n_out", 2)
    dt = episodes[0]["dt"]
    if horizons is None:
        horizons = {k: int(round(v / dt)) for k, v in HORIZON_S.items()}
    stride = max(1, int(round(stride * 0.01 / dt)))
    H = max(horizons.values())
    errs, regs = [], []
    for ep in episodes:
        F = ep["F"].copy(); F[:, 4] /= ep["long_scale"]
        reg = ep["regime_rear"] if rear else ep["regime"]
        starts = np.arange(T, len(F) - H + 1, stride)
        if len(starts) == 0:
            continue
        Ft = torch.as_tensor(F, dtype=torch.float64)
        hist = torch.stack([Ft[s - T:s] for s in starts])            # (B, T, 5)
        e = np.zeros((len(starts), H, n))
        for k in range(H):
            pred = model(hist)                                       # 시각 s+k의 [r, Uy]
            truth = Ft[starts + k]
            e[:, k] = (pred - truth[:, :n]).numpy()
            new = torch.cat([pred, truth[:, n:]], -1)[:, None]       # 예측하지 않는 채널(Ux, 조향, 종방향)은 실제 값
            hist = torch.cat([hist[:, 1:], new], 1)
        errs.append(e)
        regs.append(np.array([reg[s:s + H].max() for s in starts]))
    E, R = np.concatenate(errs), np.concatenate(regs)
    allF = np.concatenate([ep["F"][:, :n] for ep in episodes])
    std = allF.std(0)
    res = {}
    for k, m in _bins(R).items():
        if m.sum() == 0:
            continue
        res[k] = {"n": int(m.sum())}
        for name, h in horizons.items():
            x = E[m, h - 1]
            bad = ~np.isfinite(x).all(1) | (np.abs(x) > 1e3 * std).any(1)
            raw = np.sqrt(np.nanmean(np.where(bad[:, None], np.nan, x) ** 2, 0))   # 원단위 RMSE [rad/s, m/s(, m/s)]
            rm = raw / std
            res[k][name] = dict(r=float(rm[0]), uy=float(rm[1]), mean=float(rm[:2].mean()), diverged=int(bad.sum()),
                                raw=[float(v) for v in raw],
                                **({"ux": float(rm[2])} if n == 3 else {}))
    return res


class Hold:
    """기준선: 지금 값을 그대로 유지한다고 예측(팀원 평가의 "현재값 유지"). 학습 없음."""

    def __init__(self, T=4, n_out=2):
        self.T, self.n_out = T, n_out

    def __call__(self, hist):
        return hist[:, -1, :self.n_out]
