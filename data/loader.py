"""생성된 D3 데이터 로더.

  from data.loader import load_split, spielberg_windows
  eps = load_split("data/generated/f1tenth_v1", "train")
  X, Y = spielberg_windows(eps, m_nominal=5.1)
"""
import csv, json, os
import numpy as np


def load_manifest(root):
    with open(os.path.join(root, "manifest.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("mu", "k", "coupling", "mass_scale", "stiffness_scale", "frac_saturated"):
            if k in r:
                r[k] = float(r[k])
        for k in ("episode", "n_samples", "bang"):
            if k in r:
                r[k] = int(r[k])
    return rows


def load_config(root):
    with open(os.path.join(root, "config.json")) as f:
        return json.load(f)


def load_episode(root, row):
    with np.load(os.path.join(root, "episodes", row["file"])) as z:
        ep = {k: z[k] for k in z.files}
    ep["meta"] = row
    return ep


def load_split(root, split, min_samples=10):
    return [load_episode(root, r) for r in load_manifest(root) if r["split"] == split and r["n_samples"] >= min_samples]


def spielberg_windows(episodes, T=4, m_nominal=None, use_meas=True):
    """Spielberg 2019 형식: 입력 T시점 x [r, Uy, Ux, delta, Fxf/(m*100)], 타깃 다음 시점 [r, Uy].
    use_meas=True면 r, Uy, Ux는 잡음 있는 측정값을 쓴다(타깃도 측정값)."""
    Xs, Ys = [], []
    for ep in episodes:
        g = (lambda c: ep["meas_" + c]) if use_meas and "meas_r" in ep else (lambda c: ep[c])
        norm = (m_nominal * 100.0) if m_nominal else 1.0
        F = np.stack([g("r"), g("uy"), g("ux"), ep["delta"], ep["fxf"] / norm], 1)
        n = len(F) - T
        if n <= 0:
            continue
        idx = np.arange(n)[:, None] + np.arange(T)[None, :]
        Xs.append(F[idx].reshape(n, -1))
        Ys.append(F[T:, :2])
    return np.concatenate(Xs), np.concatenate(Ys)
