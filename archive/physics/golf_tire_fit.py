"""D7에서 축별 타이어 횡력을 직접 계산해 Golf 타이어 곡선(Pacejka)을 맞춘다.

golf_sysid.py(롤아웃 손실)로 맞춘 시뮬레이터는 선형 구간은 잘 맞지만 한계 근처 횡가속도를 약 13% 낮게 냈다
(25 m/s, 조향 0.08 rad: 실제 0.94 g, 시뮬레이터 0.82 g). 롤아웃 손실은 대부분이 선형 구간이라 포화 영역을 덜 본다.
그래서 Spielberg B1처럼 힘-슬립 관계를 직접 맞춘다.

  1. 운동방정식으로 축별 횡력을 푼다 (질량, Iz, a, b는 physics.params.GOLF):
       m (dUy/dt + r Ux) = Fyf cos δ + Fyr,   Iz dr/dt = a Fyf cos δ − b Fyr
  2. 슬립각: α_f = atan((Uy + a r)/Ux) − δ, α_r = atan((Uy − b r)/Ux). Uy는 센서 위치 보정(GOLF_VY_SENSOR_OFFSET) 후.
  3. 수직 하중은 정적 + 종방향 하중 이동(P2와 같은 식). 축별로 Fy/Fz = −D sin(C atan(B α)) 형태를 맞춘다.
     우리 Pacejka 구현(physics.tires.pacejka_lat)과 같은 매개화: c_alpha(=BCD/μ... 정규화 강성), C, μ.

  python -m physics.golf_tire_fit      # 화면에 맞춘 값과 정상 선회 비교, docs/figures/golf_tire_fit.png
"""
import json, os
from dataclasses import replace
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from physics.params import GOLF, GOLF_MU, GOLF_VY_SENSOR_OFFSET, TireParams
from physics.tires import pacejka_lat
from physics.golf_sysid import load_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def axle_samples(segs, p, ds):
    rows = []
    for F in segs:
        r, uy_m, ux, d, fx = (uniform_filter1d(F[:, i], 7) for i in range(5))
        uy = uy_m - ds * r
        rd, uyd = np.gradient(r, 0.01), np.gradient(uy, 0.01)
        m, a, b, L, Iz = p.m, p.a, p.b, p.L, p.Iz
        lat = m * (uyd + r * ux)
        yaw = Iz * rd
        fyf_c = (yaw + b * lat) / L             # Fyf cos δ
        fyr = (a * lat - yaw) / L
        fyf = fyf_c / np.cos(d)
        ax = fx / m
        fzf = m * 9.81 * b / L - p.h * m * ax / L
        fzr = m * 9.81 * a / L + p.h * m * ax / L
        af = np.arctan((uy + a * r) / ux) - d
        ar = np.arctan((uy - b * r) / ux)
        ok = ux > 10
        rows.append(np.c_[af, fyf / fzf, ar, fyr / fzr][ok])
    return np.vstack(rows)


def fit_axle(alpha, ynorm, tire0, mu0, iters=600):
    a = torch.as_tensor(alpha); y = torch.as_tensor(ynorm)
    th = torch.tensor([np.log(tire0.c_alpha), np.log(tire0.C), np.log(mu0)], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.02)
    for _ in range(iters):
        ca, C, mu = torch.exp(th)
        pred = pacejka_lat(a, torch.ones_like(a), mu, TireParams(c_alpha=ca, C=torch.clamp(C, 1.0, 1.99)))
        loss = torch.nn.functional.huber_loss(pred, y, delta=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    ca, C, mu = (float(v) for v in torch.exp(th))
    return dict(c_alpha=ca, C=min(max(C, 1.0), 1.99), mu=mu, loss=float(loss))


def main():
    p = GOLF
    S = axle_samples(load_runs("train"), p, GOLF_VY_SENSOR_OFFSET)
    print(f"표본 {len(S)}개. |α_f| 99분위 {np.degrees(np.percentile(np.abs(S[:, 0]), 99)):.1f}°, |Fyf/Fz| 99분위 {np.percentile(np.abs(S[:, 1]), 99):.2f}, "
          f"|α_r| 99분위 {np.degrees(np.percentile(np.abs(S[:, 2]), 99)):.1f}°, |Fyr/Fz| 99분위 {np.percentile(np.abs(S[:, 3]), 99):.2f}")
    f = fit_axle(S[:, 0], S[:, 1], p.tire_f, GOLF_MU)
    r = fit_axle(S[:, 2], S[:, 3], p.tire_r, GOLF_MU)
    print("앞:", {k: round(v, 3) for k, v in f.items()}, " 뒤:", {k: round(v, 3) for k, v in r.items()})
    json.dump(dict(front=f, rear=r, note="physics/golf_tire_fit.py, D7 학습 분할"), open(os.path.join(ROOT, "physics", "golf_tire_fit.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (ia, iy, fit, tire0, name) in zip(axs, ((0, 1, f, p.tire_f, "앞축"), (2, 3, r, p.tire_r, "뒤축"))):
        ax.plot(np.degrees(S[::7, ia]), S[::7, iy], ",", color="#999999", alpha=0.3, label="D7 (운동방정식으로 계산)")
        al = torch.linspace(-0.15, 0.15, 300, dtype=torch.float64)
        for tp, mu, lab, c in ((tire0, GOLF_MU, "롤아웃 손실로 맞춘 값 (GOLF)", "#1f77b4"),
                              (TireParams(c_alpha=fit["c_alpha"], C=fit["C"]), fit["mu"], "힘-슬립 직접 맞춤", "#d62728")):
            ax.plot(np.degrees(al), pacejka_lat(al, torch.ones_like(al), torch.tensor(mu), tp).numpy(), color=c, lw=2, label=lab)
        ax.set_xlim(-9, 9); ax.set_ylim(-1.4, 1.4); ax.grid(alpha=0.3)
        ax.set_xlabel("슬립각 [°]"); ax.set_ylabel("횡력 / 수직하중"); ax.set_title(name); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs", "figures", "golf_tire_fit.png"), dpi=120)

    # 정상 선회 비교 (25 m/s)
    from physics import P2, rk4_step
    for tag, pp, mu in (("GOLF (롤아웃)", p, GOLF_MU),
                        ("타이어 직접 맞춤", replace(p, tire_f=TireParams(c_alpha=f["c_alpha"], C=f["C"]), tire_r=TireParams(c_alpha=r["c_alpha"], C=r["C"])),
                         None)):
        out = []
        for d in (0.01, 0.02, 0.04, 0.06, 0.08):
            x = torch.tensor([[0, 0, 0, 25.0, 0, 0]], dtype=torch.float64)
            m_ = torch.tensor([mu if mu is not None else min(f["mu"], r["mu"])], dtype=torch.float64)
            for _ in range(400):
                x = rk4_step(x, torch.tensor([[d, 300.0]], dtype=torch.float64), pp, m_, 0.01, P2)
            out.append(round((x[0, 3] * x[0, 5]).item() / 9.81, 2))
        print(f"정상 선회 25 m/s, 조향 0.01/0.02/0.04/0.06/0.08 → 횡가속도 g: {tag:16s} {out}")
    print("                                                         D7 실제            [0.14, 0.28, 0.54, 0.82, 0.94]")


if __name__ == "__main__":
    main()
