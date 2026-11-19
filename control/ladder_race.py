"""팀원 방식 제어 주행(iac_sim.py)을 Golf / Thunderhill West 로 옮긴 것. 결과는 viz/ladder_race.py 로 영상.

세계(실제 차 역할): 우리 Golf 시뮬레이터(physics GOLF 프리셋, 경사 포함, μ 1.085). 사다리 모델과 따로 만든 것이라
  어느 모델 편도 들지 않는다. (팀원은 학습 모델 하나를 세계로 썼다.)
제어기: 팀원 MPPI 그대로 (K 1024 후보, N 25 스텝 = 1초, 같은 비용·잡음·가중 평균). 비교 모델만 바꾼다.
경로: D7 학습 주행 중 가장 빠른 한 바퀴의 실제 주행선 + 그때 속도(목표 속도). 팀원 iac_track.py 와 같은 방식.

  python -m control.ladder_race track                          # -> data/tracks/thunderhill_race.npz
  python -m control.ladder_race run --exp base --rung R2       # -> eval/results/ladder_race/base__R2_s0.{json,npz}
  python -m control.ladder_race run --rung SIM                 # 정답 모델(세계 자신)로 제어 = MPPI 자체 점검
  python -m control.ladder_race run --rung R2 --scale 1.06     # 한계 시험: D7 기록 속도 × 1.06 -> base__R2_s0__x1.06.*
  (D7 기준 바퀴는 이미 횡 1.07 g(99분위)로 코너는 한계 근처, 제동만 0.45 g로 약하다.
   모드 L(곡률·마찰로 계산한 한계 속도 프로파일)은 강한 제동 + 시케인에서 정답 모델도 스핀해 쓰지 않는다.)

Golf 에 맞춘 것 (팀원 원본 대비)
  조작 u = [종방향 a ∈ [-1, 1], 조향 δ]. 모델 입력 lon(앞바퀴 힘 N) = a·8000 (구동) / a·10000 (제동),
    구동은 출력 150 kW 제한(lon ≤ P/vx). D7 범위: lon −5600~+7900 N. 잡음·가중치는 팀원 값을 그대로 쓰려고 정규화했다.
  조향 한계 ±0.35 rad (D7 −0.19~+0.33), 조향 변화 스텝당 0.03 rad (D7 99.9분위 0.63 rad/s).
  트랙 폭이 좁아(약 8.8 m, AV-21 트랙의 절반 이하) 이탈 판정을 '경계 + 1 m'로 (원본 + 2 m), 비용의 경계 여유를 0 m 로 (원본 1.5 m).
  입력 행 = [vx, vy, r, lon, δ, 기울기, 경사] (우리 방식 노면). 세계에는 경사만 넣는다 (기울기 이득은 시뮬레이터 맞춤에서 믿기 어려웠음).
"""
import argparse, glob, json, math, os, sys, time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "models", "ladder"), ROOT]
import ladder, configs                                                    # noqa: E402
from data.datasets import D7_DIR, D7_DUP, D7_SPLIT                        # noqa: E402
from data.ladder_data import TRACK                                        # noqa: E402
from physics.bicycle import rk4_step                                      # noqa: E402
from physics.params import GOLF, GOLF_MU, GOLF_VY_SENSOR_OFFSET           # noqa: E402

DEV = torch.device("cpu")
DT, H = 0.04, 8
TRACK_NPZ = os.path.join(ROOT, "data", "tracks", "thunderhill_race.npz")
CKPT = os.path.join(ROOT, "checkpoints", "ladder")
OUT = os.path.join(ROOT, "eval", "results", "ladder_race")
F_DRIVE, F_BRAKE, P_MAX = 8000.0, 10000.0, GOLF.power_max   # 제동: 앞 10000 N = 전체 약 0.83 g (D7 기록은 앞 −5600 N까지라 모델에게는 범위 밖)
U_LO = torch.tensor([-1.0, -0.35])
U_HI = torch.tensor([1.0, 0.35])
DSTEER = 0.03
OFF_FAIL, OFF_COST = 1.0, 0.0
MU_CAP = GOLF_MU               # 모드 B 속도 상한에 쓰는 μ = 세계의 실제 μ (1.25로 하면 1초 계획으로는 정답 모델도 코너를 못 돌았다)
OFF_COST_B = 0.8              # 모드 B: 무게중심이 경계에서 0.8 m 안쪽 (차 반폭 0.9 m → 바퀴가 경계선 근처까지)
SPIN = math.radians(30)


def lon_of(a, vx):
    """정규화 종방향 a → 앞바퀴 힘 [N] (구동은 출력 제한)."""
    drive = torch.minimum(a * F_DRIVE, P_MAX / torch.clamp(vx, min=1.0))
    return torch.where(a >= 0, drive, a * F_BRAKE)


# ---------------------------------------------------------------- 트랙 (팀원 iac_track.py 방식)
def build_track():
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial import cKDTree
    from data.thunderhill_west import resample_closed
    best = None
    for f in sorted(glob.glob(os.path.join(D7_DIR, "*.csv"))):
        name = os.path.basename(f)[:-4]
        if name in D7_DUP or any(name in v for v in D7_SPLIT.values()):
            continue                                                      # 학습 주행만
        a = pd.read_csv(f, comment="#", usecols=["posE_m", "posN_m", "Vx_mps"])[["posE_m", "posN_m", "Vx_mps"]].to_numpy()
        a = a[:len(a) // 8 * 8].reshape(-1, 8, 3).mean(1)
        x, y, v = a.T
        d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        for i0 in range(0, len(x), 25):                                   # 출발점을 1초씩 옮겨 가며 (정지 출발 구간 제외)
            far = np.where(d - d[i0] > 1500)[0]
            if not len(far):
                break
            j = far[0]
            k = np.where(np.hypot(x[j:] - x[i0], y[j:] - y[i0]) < 1.5)[0]
            if not len(k):
                continue
            k = j + k[0]
            if v[i0:k].min() > 5 and (best is None or v[i0:k].mean() > best[0]):
                best = (v[i0:k].mean(), x[i0:k], y[i0:k], v[i0:k], name)
    _, lx, ly, lv, lname = best
    mid = resample_closed(np.c_[lx, ly], 1.0, sigma=2.0)
    dl = np.r_[0, np.cumsum(np.hypot(np.diff(lx), np.diff(ly)))]
    n = len(mid)
    v = uniform_filter1d(np.interp(np.arange(n) * dl[-1] / n, dl, lv), 15, mode="wrap")
    sh = int(np.argmax(uniform_filter1d(v, 51, mode="wrap")))           # 출발점 = 가장 빠른 직선
    mid, v = np.roll(mid, -sh, 0), np.roll(v, -sh)
    d = np.gradient(mid, axis=0)
    psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    nrm = np.stack([-np.sin(psi), np.cos(psi)], 1)
    # 폭·노면: 트랙 파일 중심선(경계 = 중심 ± 폭)에서 가져온다
    T = np.loadtxt(TRACK, delimiter=",", comments="#")
    c = T[:, :2]
    dc = np.gradient(np.r_[c[-1:], c, c[:1]], axis=0)[1:-1]
    pc = np.arctan2(dc[:, 1], dc[:, 0])
    nc = np.stack([-np.sin(pc), np.cos(pc)], 1)
    j = cKDTree(c).query(mid)[1]
    same = np.einsum("ij,ij->i", d / np.linalg.norm(d, axis=1, keepdims=True), np.stack([np.cos(pc[j]), np.sin(pc[j])], 1))
    assert np.median(same) > 0.9, "트랙 파일 방향이 주행 방향과 반대"
    o = np.einsum("ij,ij->i", mid - c[j], nc[j])                         # 중심선 기준 왼쪽 + 오프셋
    wl, wr = T[j, 3] - o, T[j, 2] + o
    left, right = c + nc * T[:, 3:4], c - nc * T[:, 2:3]
    np.savez(TRACK_NPZ, x=mid[:, 0], y=mid[:, 1], psi=psi, width_l=wl, width_r=wr, vref=np.maximum(v, 8.0),
             bank=T[j, 6], grade=T[j, 5], left=left, right=right, center=c)
    print(f"기준 바퀴 {lname}: 길이 {n} m, 평균 {lv.mean():.1f} m/s, vref {v.min():.1f}~{v.max():.1f}, "
          f"경계까지 최소 왼 {wl.min():.2f} / 오 {wr.min():.2f} m -> {os.path.relpath(TRACK_NPZ, ROOT)}")


def speed_cap(x, y, mu=MU_CAP, v_top=50.0, brake_g=0.6):
    """모드 B 속도 상한(한계 속도 프로파일): 곡률 한계 sqrt(μg/|κ|) + 제동 거리(감속 brake_g)로 앞 코너를 미리 반영.
    1초 계획이 40 m 앞까지만 봐서 생기는 '늦은 제동'을 막는 장치 (Deep Dynamics 등이 MPC에 한계 속도를 주는 방식)."""
    from scipy.ndimage import uniform_filter1d
    p = np.c_[x, y]
    d = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / 2
    dd = np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)
    k = uniform_filter1d((d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]) / np.hypot(*d.T) ** 3, 9, mode="wrap")
    v = np.minimum(np.sqrt(mu * 9.81 / np.maximum(np.abs(k), 1e-4)), v_top)
    n = len(v)
    for _ in range(3):                                                    # 뒤에서 앞으로: 제동 거리
        for i in range(2 * n - 1, -1, -1):
            v[i % n] = min(v[i % n], math.sqrt(v[(i + 1) % n] ** 2 + 2 * brake_g * 9.81))
    return v


class Track:
    def __init__(self):
        t = np.load(TRACK_NPZ)
        self.np = {k: t[k] for k in t.files}
        self.n = len(t["x"])
        f = lambda k: torch.tensor(t[k], dtype=torch.float32)
        self.x, self.y, self.psi, self.vref = f("x"), f("y"), f("psi"), f("vref")
        self.wl, self.wr, self.bank, self.grade = f("width_l"), f("width_r"), f("bank"), f("grade")
        self.set_cap(1.0)

    def project(self, X, Y, idx0, back=10, ahead=90):
        off = torch.arange(-back, ahead)
        cand = (idx0.unsqueeze(1) + off) % self.n
        d2 = (self.x[cand] - X.unsqueeze(1)) ** 2 + (self.y[cand] - Y.unsqueeze(1)) ** 2
        idx = cand.gather(1, d2.argmin(1, keepdim=True)).squeeze(1)
        psi = self.psi[idx]
        ey = -(X - self.x[idx]) * torch.sin(psi) + (Y - self.y[idx]) * torch.cos(psi)
        return idx, ey

    def set_cap(self, frac):
        """속도 상한 μ = frac × 세계 μ."""
        self.vcap = torch.tensor(speed_cap(self.np["x"], self.np["y"], mu=frac * MU_CAP), dtype=torch.float32)

    def road(self, ix):
        return torch.stack([self.bank[ix], self.grade[ix]], -1)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def kin(pose, s):
    X, Y, psi = pose.unbind(-1)
    vx, vy, r = s.unbind(-1)
    pm = psi + 0.5 * DT * r
    return torch.stack([X + DT * (vx * torch.cos(pm) - vy * torch.sin(pm)),
                        Y + DT * (vx * torch.sin(pm) + vy * torch.cos(pm)), psi + DT * r], -1)


# ---------------------------------------------------------------- 세계 = Golf 시뮬레이터
class SimModel(torch.nn.Module):
    """사다리 모델과 같은 꼴: hist (B, 8, 7) → 다음 [vx, vy_측정, r]. 정답 모델(SIM)로도 쓴다."""

    def forward(self, hist):
        row = hist[:, -1]
        vx, vym, r, lon, delta, bank, grade = row.unbind(-1)
        vy = vym - GOLF_VY_SENSOR_OFFSET * r
        x = torch.stack([torch.zeros_like(vx)] * 3 + [vx, vy, r], -1)
        x = self.step(x, lon, delta, grade)
        return torch.stack([x[:, 3], x[:, 4] + GOLF_VY_SENSOR_OFFSET * x[:, 5], x[:, 5]], -1)

    @staticmethod
    def step(x, lon, delta, grade):
        fx = torch.where(lon >= 0, lon, lon / GOLF.brake_front)          # 앞바퀴 힘 → 전체 힘 명령
        u = torch.stack([delta, fx], -1)
        slope = torch.stack([grade, torch.zeros_like(grade)], -1)
        for _ in range(4):
            x = rk4_step(x, u, GOLF, GOLF_MU, DT / 4, slope=slope)
        return x


# ---------------------------------------------------------------- MPPI (팀원 iac_sim.py 그대로, 입력만 Golf 형식)
class MPPI:
    def __init__(self, model, track, K=1024, N=25, sigma=(0.12, 0.06), lam=0.3, seed=0, corr=0.85, mode="A"):
        self.m, self.tr, self.K, self.N, self.corr, self.lam, self.mode = model, track, K, N, corr, lam, mode
        self.sig = torch.tensor(sigma)
        self.U = torch.zeros(N, 2)
        self.U[:, 0] = 0.2
        self.g = torch.Generator().manual_seed(seed)
        self.w = dict(ey=1.0, epsi=50.0, v=0.5, du_d=2e4, du_t=5.0, off=1e4, bad=1e6, slow=200.0)
        if mode == "B":   # 한계 주행: 목표 속도 대신 진행 거리 보상, 경로 오차는 약하게 (트랙 폭 안에서 자유롭게)
            self.w.update(ey=0.1, epsi=20.0, v=0.0, prog=20.0, cap=50.0, beta=2e3)

    @torch.no_grad()
    def rollout(self, U, hist, s, P, ix, up):
        U = torch.max(torch.min(U, U_HI), U_LO)
        cost = torch.zeros(self.K)
        for k in range(self.N):
            u = U[:, k]
            row = torch.cat([s, lon_of(u[:, 0], s[:, 0]).unsqueeze(1), u[:, 1:2], self.tr.road(ix)], 1)
            h = torch.cat([hist, row.unsqueeze(1)], 1)
            s_next = self.m(h)
            bad = ~torch.isfinite(s_next).all(1) | (s_next.abs() > 200).any(1)
            s_next = torch.where(bad.unsqueeze(1), s, s_next)
            P = kin(P, s_next)
            ix, ey = self.tr.project(P[:, 0], P[:, 1], ix)
            epsi = wrap(P[:, 2] - self.tr.psi[ix])
            half = torch.where(ey > 0, self.tr.wl[ix], self.tr.wr[ix])
            off = (ey.abs() > half - (OFF_COST_B if self.mode == "B" else OFF_COST)).float()
            du = u - up
            if self.mode == "B":
                ds = DT * (s_next[:, 0] * torch.cos(epsi) - s_next[:, 1] * torch.sin(epsi))
                beta = torch.atan2(s_next[:, 1], s_next[:, 0].clamp(min=1.0))
                cost = cost - self.w["prog"] * ds + self.w["cap"] * torch.relu(s_next[:, 0] - self.tr.vcap[ix]) ** 2                     + self.w["beta"] * torch.relu(beta.abs() - 0.15) ** 2
            cost = cost + (self.w["ey"] * ey ** 2 + self.w["epsi"] * epsi ** 2 + self.w["v"] * (s_next[:, 0] - self.tr.vref[ix]) ** 2
                           + self.w["du_d"] * du[:, 1] ** 2 + self.w["du_t"] * du[:, 0] ** 2 + self.w["off"] * off
                           + self.w["bad"] * bad.float() + self.w["slow"] * torch.relu(6.0 - s_next[:, 0]) ** 2)
            up, hist, s = u, h[:, 1:], s_next
        return U, cost

    @torch.no_grad()
    def act(self, hist_past, s_now, pose, idx, u_prev):
        K, N = self.K, self.N
        wn = torch.randn(K, N, 2, generator=self.g)
        eps = torch.empty_like(wn)
        eps[:, 0] = wn[:, 0]
        c = self.corr
        for k in range(1, N):
            eps[:, k] = c * eps[:, k - 1] + (1 - c * c) ** 0.5 * wn[:, k]
        sig = self.sig.clone()
        sig[1] = sig[1] * min(1.0, max(0.1, 6.0 / max(float(s_now[0]), 1.0)))
        eps = eps * sig
        eps[0] = 0.0
        U, cost = self.rollout(self.U.unsqueeze(0) + eps, hist_past.unsqueeze(0).expand(K, -1, -1),
                               s_now.unsqueeze(0).expand(K, 3), pose.unsqueeze(0).expand(K, 3),
                               torch.full((K,), int(idx), dtype=torch.long), u_prev.unsqueeze(0).expand(K, 2))
        c = cost - cost.min()
        wts = torch.softmax(-c / (self.lam * (c.median() + 1e-6)), 0)
        self.U = (wts[:, None, None] * U).sum(0)
        u0 = self.U[0].clone()
        u0[1] = u_prev[1] + torch.clamp(u0[1] - u_prev[1], -DSTEER, DSTEER)
        self.U[0] = u0
        self.U = torch.cat([self.U[1:], self.U[-1:]], 0)
        return u0


@torch.no_grad()
def episode(ctrl, tr, K=1024, N=25, seed=0, max_time=None, verbose=True, mode="A"):
    v0 = min(float(tr.vref[0]), 32.0)
    x = torch.tensor([[float(tr.x[0]), float(tr.y[0]), float(tr.psi[0]), v0, 0.0, 0.0]])   # 시뮬레이터 상태
    s = torch.tensor([v0, 0.0, 0.0])
    u_prev = torch.tensor([0.15, 0.0])
    row0 = torch.cat([s, lon_of(u_prev[:1], s[:1]), u_prev[1:], tr.road(torch.tensor(0))])
    hist_past = row0.repeat(H - 1, 1)
    mp = MPPI(ctrl, tr, K=K, N=N, seed=seed, mode=mode)
    idx = torch.zeros(1, dtype=torch.long)
    max_time = max_time or 2.0 * float((1.0 / tr.vref).sum())
    prog, last_i, status, t0 = 0, 0, "timeout", time.time()
    log = {k: [] for k in ("X", "Y", "psi", "vx", "vy", "r", "a", "lon", "delta", "ey", "vref", "idx")}
    for t in range(int(max_time / DT)):
        pose = x[0, :3]
        u = mp.act(hist_past, s, pose, idx, u_prev)
        u = torch.max(torch.min(u, U_HI), U_LO)
        lon = lon_of(u[:1], s[:1])
        row = torch.cat([s, lon, u[1:], tr.road(idx[0])])
        x = SimModel.step(x, lon, u[1:], tr.grade[idx])
        if not torch.isfinite(x).all():
            status = "world_nan"
            break
        s_next = torch.stack([x[0, 3], x[0, 4] + GOLF_VY_SENSOR_OFFSET * x[0, 5], x[0, 5]])
        idx, ey = tr.project(x[:, 0], x[:, 1], idx)
        i = int(idx)
        di = (i - last_i) % tr.n
        if di < tr.n / 2:
            prog += di
        last_i = i
        half = float(tr.wl[i] if ey > 0 else tr.wr[i])
        for k, v in zip(log, [x[0, 0], x[0, 1], x[0, 2], s_next[0], s_next[1], s_next[2], u[0], lon[0], u[1], ey[0], tr.vref[i], idx[0]]):
            log[k].append(float(v))
        hist_past = torch.cat([hist_past[1:], row.unsqueeze(0)], 0)
        s, u_prev = s_next, u
        if verbose and t % 250 == 0:
            print(f"  t {t * DT:5.1f} s  진행 {prog / tr.n * 100:5.1f}%  vx {float(s[0]):5.1f}  ey {float(ey):+5.2f}  "
                  f"({time.time() - t0:.0f} s 경과)", flush=True)
        if abs(float(ey)) > half + OFF_FAIL:
            status = "off_track"
            break
        if abs(math.atan2(float(x[0, 4]), float(x[0, 3]))) > SPIN:
            status = "spin"
            break
        if float(s[0]) < 3.0:
            status = "stalled"
            break
        if prog >= tr.n - 2:
            status = "finished"
            break
    T = len(log["ey"]) * DT
    ey = np.abs(np.array(log["ey"]))
    res = dict(status=status, finished=status == "finished", progress_m=float(prog), progress_frac=float(prog / tr.n),
               time_s=T, lap_time=T if status == "finished" else None, ey_mean=float(ey.mean()), ey_max=float(ey.max()),
               v_err_mean=float(np.mean(np.abs(np.array(log["vx"]) - np.array(log["vref"])))),
               v_mean=float(np.mean(log["vx"])), v_max=float(np.max(log["vx"])), wall_s=time.time() - t0, K=K, N=N, seed=seed, mode=mode)
    return res, log


def load_ctrl(exp, rung, seed):
    if rung == "SIM":
        return SimModel()
    p = os.path.join(CKPT, exp, f"{rung}_s{seed}.pt")
    s = json.load(open(p[:-3] + ".json"))["spec"]
    assert s["lon"] == "force" and s["road"] == "bank+grade", "지금은 force + 기울기·경사 모델만"
    return ladder.load_model(p, configs.golf("force", True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["track", "run"])
    ap.add_argument("--exp", default="base")
    ap.add_argument("--rung", default="R2")
    ap.add_argument("--seed", type=int, default=0, help="모델 시드")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=25)
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cap", type=float, default=0.9, help="모드 B 속도 상한의 μ 비율 (1 = 세계 μ 1.085)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scale", type=float, default=1.0, help="모드 A 목표 속도 배율 (D7 기록 속도 × 배율)")
    ap.add_argument("--mode", default="A", choices=["A", "B", "L"],
                    help="A: D7 기록 속도 따라가기(팀원 방식), L: 한계 속도 프로파일 따라가기, B: 진행 거리 최대(불안정, 정답 모델도 이탈)")
    a = ap.parse_args()
    if a.cmd == "track":
        return build_track()
    torch.set_num_threads(a.threads)
    tr = Track()
    tr.set_cap(a.cap)
    if a.scale != 1.0:
        tr.vref = tr.vref * a.scale             # D7 기록 속도 × 배율 (기록이 이미 횡 1.07 g 근처라 배율이 곧 한계 초과 정도)
    if a.mode == "L":
        tr.vref = tr.vcap                       # 한계 속도 프로파일을 목표 속도로 (비용은 팀원 그대로)
    res, log = episode(load_ctrl(a.exp, a.rung, a.seed), tr, a.K, a.N, seed=0, max_time=a.max_time, mode=a.mode)
    res.update(exp=a.exp, rung=a.rung, model_seed=a.seed, cap=a.cap if a.mode != "A" else None, scale=a.scale)
    os.makedirs(OUT, exist_ok=True)
    tag = (f"{a.exp}__{a.rung}_s{a.seed}" if a.rung != "SIM" else "SIM") + ("" if a.mode == "A" else f"__{a.mode}") + (f"__x{a.scale:g}" if a.scale != 1.0 else "") + a.tag
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, tag + "_traj.npz"), **{k: np.array(v) for k, v in log.items()})
    print(tag, json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
