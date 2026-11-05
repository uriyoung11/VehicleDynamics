"""물리 모델·생성기 검증. pytest 없이도 `python tests/test_physics.py`로 돌아간다."""
import math, os, sys, tempfile
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physics import FULLSCALE, F1TENTH, P0, P1, P2, ModelOptions, forces, derivatives, rk4_step, euler_step

torch.set_default_dtype(torch.float64)


def random_states(n, p, gen):
    vlo, vhi = p.ux_target_range
    ux = vlo + (vhi - vlo) * torch.rand(n, generator=gen)
    x = torch.zeros(n, 6)
    x[:, 3] = ux
    x[:, 4] = 0.15 * ux * (2 * torch.rand(n, generator=gen) - 1)
    x[:, 5] = 9.81 / ux * (2 * torch.rand(n, generator=gen) - 1)
    u = torch.stack([p.delta_max * (2 * torch.rand(n, generator=gen) - 1),
                     p.m * 9.81 * (2 * torch.rand(n, generator=gen) - 1)], -1)
    return x, u


def test_matches_spielberg_bike_model():
    """P1(Fiala, 하중 이동·결합·저항 없음) 오일러 1스텝 = Spielberg 원논문 Bike_Model 식."""
    sys.path.insert(0, os.path.join(ROOT, "repro", "spielberg2019"))
    from repro import BikeModel, FX_NORM
    gen = torch.Generator().manual_seed(0)
    p = FULLSCALE
    x, u = random_states(2000, p, gen)
    u[:, 1] = torch.where(u[:, 1] > 0, u[:, 1], u[:, 1])  # Fx 그대로
    opt = ModelOptions(tire="fiala", load_transfer=False, combined=False, resistance=False)
    # Spielberg 입력은 앞바퀴 힘(fxf). 우리 모델에 같은 fxf가 들어가도록 전륜 100%로 둔다
    from dataclasses import replace
    pf = replace(p, drive_front=1.0, brake_front=1.0)
    ours = euler_step(x, u, pf, 1.0, 0.01, opt)[:, [5, 4]]
    bike = BikeModel(np.random.default_rng(0))
    with torch.no_grad():
        for prm in (bike.Ca_f_norm, bike.Ca_r_norm, bike.mu_norm):
            prm.fill_(1.0)
        feats = torch.zeros(len(x), 20)
        feats[:, 15:20] = torch.stack([x[:, 5], x[:, 4], x[:, 3], u[:, 0], u[:, 1] / FX_NORM], -1)
        theirs = bike(feats)
    err = (ours - theirs).abs().max().item()
    assert err < 1e-9, err


def test_friction_circle():
    gen = torch.Generator().manual_seed(1)
    for p in (FULLSCALE, F1TENTH):
        for mu in (0.2, 0.5, 1.0):
            x, u = random_states(5000, p, gen)
            f = forces(x, u, p, mu, P2)
            for ax in "fr":
                tot = torch.sqrt(f["fx" + ax] ** 2 + f["fy" + ax] ** 2)
                assert (tot <= mu * f["fz" + ax] * (1 + 1e-6)).all(), (p.name, mu, ax)


def steady_state_ay_max(p, mu, v):
    """일정 속도에서 조향을 천천히 늘려 최대 정상 횡가속도를 구한다(정상선회 램프 조향)."""
    x = torch.tensor([[0, 0, 0, v, 0, 0.0]])
    best, dt = 0.0, 0.005
    n = int(15.0 / dt)
    for i in range(n):
        d = p.delta_max * i / n
        # 저항만큼 구동해서 속도 유지
        fx = p.m * 2.0 * (v - x[0, 3]) + (p.c0 + p.c1 * x[0, 3] + p.c2 * x[0, 3] ** 2)
        u = torch.tensor([[d, float(fx)]])
        x = rk4_step(x, u, p, mu, dt, P2)
        if abs(math.atan2(x[0, 4], x[0, 3])) > math.radians(20):
            break
        _, f = derivatives(x, u, p, mu, P2, return_forces=True)
        best = max(best, abs(float(f["ay"][0])))
    return best / (mu * p.g)


def test_steady_state_lateral_limit():
    """최대 정상 횡가속도는 대략 μg (0.8~1.02) 이어야 한다."""
    for p, v in ((FULLSCALE, 15.0), (F1TENTH, 3.0)):
        for mu in (0.3, 1.0):
            r = steady_state_ay_max(p, mu, v)
            assert 0.8 < r < 1.02, (p.name, mu, r)


def test_rk4_convergence():
    gen = torch.Generator().manual_seed(2)
    p = F1TENTH
    x0, u = random_states(64, p, gen)
    u[:, 1] *= 0.2
    def run(dt):
        x, ux_min = x0.clone(), x0[:, 3].clone()
        for _ in range(int(round(1.0 / dt))):
            x = rk4_step(x, u, p, 0.7, dt, P2)
            ux_min = torch.minimum(ux_min, x[:, 3])
        return x, ux_min
    (a, ux_min), (b, _) = run(1e-3), run(5e-4)
    # 도중에 뒤로 미끄러진(ux<0.5) 경우는 슬립각 식이 성립하지 않으니 뺀다
    ok = torch.isfinite(a).all(-1) & (ux_min > 0.5)
    assert ok.sum() > 32
    assert (a[ok, 3:] - b[ok, 3:]).abs().max() < 1e-4


def test_generator_and_loader():
    from data import synth
    from data.loader import load_manifest, load_split, spielberg_windows
    with tempfile.TemporaryDirectory() as d:
        synth.main(["--vehicle", "f1tenth", "--episodes", "12", "--duration", "3", "--out", d, "--seed", "0"])
        rows = load_manifest(d)
        assert len(rows) == 12
        eps = [e for s in ("train", "val", "test", "test_mu_interp", "test_mu_extrap") for e in load_split(d, s, min_samples=5)]
        ep = eps[0]
        assert np.isfinite(ep["ux"]).all() and len(ep["t"]) == ep["meta"]["n_samples"]
        assert np.allclose(np.diff(ep["t"]), 0.01, atol=1e-5)
        X, Y = spielberg_windows(eps, m_nominal=F1TENTH.m)
        assert X.shape[1] == 20 and len(X) == len(Y)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(fails)
