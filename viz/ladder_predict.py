"""팀원 방식(iac_viz2d.py predict 모드)의 2D 탑뷰 영상을 Golf/D7/Thunderhill West 로 옮긴 것.

D7 시험 주행(학습에 안 씀)을 재생하면서, 매 순간 모델별 '1초 동안 어디로 갈지' 예측선을 차 앞에 그린다.
흰색 = 실제로 간 길, 색깔 = 모델 예측 (기록된 조작·노면을 그대로 넣은 열린 루프).
화면 구성·색은 팀원 원본과 같다: 차를 따라가는 카메라(차 머리가 위), 미니맵, 속도, 조향·구동 막대,
마찰원, 타이어 한계 사용률, 예측 끝에서 빗나간 거리 막대.

  python viz/ladder_predict.py                                  # 기준(①) R1 R2 R4 R5 -> viz/out/ladder/predict_base.mp4
  python viz/ladder_predict.py --models base/R2 d1/R2 base/R5 d1/R5 --out viz/out/ladder/predict_base_vs_d1.mp4
  python viz/ladder_predict.py --preview 300                    # 한 장면만 png

모델은 checkpoints/ladder/<실험>/<모델>_s<시드>.pt (models/ladder/run_ablation.py 로 학습한 것).
"""
import argparse, json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import animation, font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Polygon
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "models", "ladder"), ROOT]
import ladder, configs                                               # noqa: E402
from data.datasets import D7_DIR, D7_K_DRIVE, D7_K_BRAKE             # noqa: E402
from data.ladder_data import _road_lookup, TRACK                     # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
for f in ["Malgun Gothic", "NanumGothic"]:          # 팀원 원본의 NanumSquare 는 °·← 글자가 없다
    m = [x.name for x in font_manager.fontManager.ttflist if f.lower() in x.name.lower()]
    if m:
        plt.rcParams["font.family"] = m[0]
        break
plt.rcParams["axes.unicode_minus"] = False

BG, ASPH, FG, DIM = "#14161a", "#3a3e45", "#e8e8e8", "#8a8f98"
CAR_L, CAR_W = 4.26, 1.79                     # Golf
DT, H = 0.04, 8
CKPT = os.path.join(ROOT, "checkpoints", "ladder")
COL = {"R1": "#4C8DF6", "R2": "#2FBF71", "R3": "#3CC6C6", "R4": "#B07CF2", "R5": "#F5883A", "R6": "#E05A5A"}
COL2 = {"R1": "#A9C8FB", "R2": "#9BE6BD", "R3": "#9DE3E3", "R4": "#D9C0FA", "R5": "#FAC49C", "R6": "#F0A9A9"}  # 두 번째 실험(옅은 색)
KO = {"R1": "물리", "R2": "물리+보정", "R3": "신경망 타이어", "R4": "DD식", "R5": "신경망", "R6": "블랙박스"}
EXP_KO = {"base": "D7만", "d1": "+D1", "pedal": "페달", "road_none": "노면 없음", "road_bank": "기울기만", "step1": "1스텝"}
RUN_KO = {"2023-10_4": "D7 시험 주행 2023-10_4", "2023-11_3": "D7 시험 주행 2023-11_3", "2023-11_8": "D7 시험 주행 2023-11_8"}


# ---------------------------------------------------------------- 데이터
def load_run(run):
    """D7 주행 하나를 25 Hz 로: 상태·입력(force, 기울기+경사)·위치·차 방향. vx > 5 인 가장 긴 구간."""
    cols = ["Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad", "engineTorque_Nm", "GearRatio",
            "brake_fl_N", "brake_fr_N", "posE_m", "posN_m", "Ve_mps", "Vn_mps", "axCG_mps2", "ayCG_mps2"]
    a = pd.read_csv(os.path.join(D7_DIR, run + ".csv"), comment="#", usecols=cols)[cols].to_numpy(np.float64)
    n = len(a) // 8 * 8
    a = a[:n].reshape(-1, 8, a.shape[1]).mean(1)
    vx, vy, r, delta = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    lon = D7_K_DRIVE / D7_K_BRAKE * a[:, 4] * a[:, 5] - (a[:, 6] + a[:, 7])
    road = _road_lookup()(a[:, 8:10])                              # [bank, grade]
    # 차 방향 = 이동 방향 − 미끄럼각 (팀원 영상과 같은 방식)
    head = np.unwrap(np.arctan2(a[:, 11], a[:, 10])) - np.arctan2(vy, vx)
    ok = vx > 5
    e = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    i, j = max(zip(e[::2], e[1::2]), key=lambda p: p[1] - p[0])
    sl = slice(i, j)
    return dict(s=np.c_[vx, vy, r][sl], lon=lon[sl], delta=delta[sl], bank=road[sl, 0], grade=road[sl, 1],
                x=a[sl, 8], y=a[sl, 9], head=head[sl], ax=a[sl, 12], ay=a[sl, 13])


def u_of(sg, road):
    if road == "bank+grade":
        return np.c_[sg["lon"], sg["delta"], sg["bank"], sg["grade"]]
    return np.c_[sg["lon"], sg["delta"], sg["bank"] if road == "bank" else 0 * sg["bank"]]


class Track:
    def __init__(self):
        T = np.loadtxt(TRACK, delimiter=",", comments="#")
        c = T[:, :2]
        d = np.gradient(np.r_[c[-1:], c, c[:1]], axis=0)[1:-1]
        psi = np.arctan2(d[:, 1], d[:, 0])
        nrm = np.stack([-np.sin(psi), np.cos(psi)], 1)
        self.c, self.left, self.right = c, c + nrm * T[:, 3:4], c - nrm * T[:, 2:3]

    def near(self, p, rad):
        return np.where(np.hypot(*(self.c - p).T) < rad)[0]


# ---------------------------------------------------------------- 그리기 (팀원 원본과 같음)
def to_cam(P, p0, psi0):
    d = np.asarray(P) - p0
    a = math.pi / 2 - psi0
    ca, sa = math.cos(a), math.sin(a)
    return np.stack([ca * d[..., 0] - sa * d[..., 1], sa * d[..., 0] + ca * d[..., 1]], -1)


def car_poly(x, y, psi):
    pts = np.array([[CAR_L / 2, CAR_W / 2], [CAR_L / 2 + 0.5, 0], [CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, -CAR_W / 2],
                    [-CAR_L / 2, CAR_W / 2]])
    c, s = math.cos(psi), math.sin(psi)
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)


def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


class MainView:
    def __init__(self, fig, track, rect, zoom=(12.0, 1.0)):
        self.zoom, self.track = zoom, track
        self.ax = fig.add_axes(rect)
        style_ax(self.ax)
        self.ax.set_aspect("equal")
        self.road = PolyCollection([], facecolor=ASPH, edgecolor=ASPH, lw=0.6, zorder=1)
        self.ax.add_collection(self.road)
        self.bl, = self.ax.plot([], [], color=FG, lw=1.6, zorder=2)
        self.br, = self.ax.plot([], [], color=FG, lw=1.6, zorder=2)
        self.cl, = self.ax.plot([], [], color="#6b7079", lw=0.8, ls=(0, (6, 6)), zorder=2)
        self.halfw = None
        fw, fh = fig.get_size_inches()
        self.aspect = (rect[2] * fw) / (rect[3] * fh)

    def frame(self, p0, psi0, vx):
        target = self.zoom[0] + self.zoom[1] * max(vx, 0.0)
        self.halfw = target if self.halfw is None else self.halfw + 0.04 * (target - self.halfw)
        hw = self.halfw
        idx = self.track.near(p0, 2.2 * hw * max(1, self.aspect))
        L, R, C = (to_cam(a[idx], p0, psi0) for a in (self.track.left, self.track.right, self.track.c))
        brk = np.where(np.diff(idx) != 1)[0] + 1
        with_nan = lambda A: np.insert(A, brk, np.nan, axis=0) if len(brk) else A
        quads = [np.array([L[i], L[i + 1], R[i + 1], R[i]]) for i in range(len(idx) - 1) if idx[i + 1] - idx[i] == 1]
        if len(idx) > 1 and idx[0] == 0 and idx[-1] == len(self.track.c) - 1:
            quads.append(np.array([L[-1], L[0], R[0], R[-1]]))
        self.road.set_verts(quads)
        for ln, A in ((self.bl, L), (self.br, R), (self.cl, C)):
            A = with_nan(A)
            ln.set_data(A[:, 0], A[:, 1])
        cy = 0.45 * hw
        self.ax.set_xlim(-hw * self.aspect, hw * self.aspect)
        self.ax.set_ylim(cy - hw, cy + hw)


class MiniMap:
    def __init__(self, fig, track, rect):
        self.ax = fig.add_axes(rect)
        style_ax(self.ax)
        self.ax.set_aspect("equal")
        self.ax.plot(*np.r_[track.c, track.c[:1]].T, color=DIM, lw=3)

    def add(self, color, size=7):
        return self.ax.plot([], [], "o", color=color, ms=size, mec="k", mew=0.5, zorder=5)[0]


# ---------------------------------------------------------------- 영상
def main(a):
    NP = int(round(a.horizon / DT))
    sg = load_run(a.run)
    n = len(sg["s"])
    k0 = max(H, int(a.t0 / DT))
    k1 = n - NP - 1 if not a.dur else min(n - NP - 1, k0 + int(a.dur / DT))
    ks = np.arange(k0, k1)
    pose0 = np.stack([sg["x"][ks], sg["y"][ks], sg["head"][ks]], 1)
    truth = np.stack([np.stack([sg["x"][k + 1:k + 1 + NP], sg["y"][k + 1:k + 1 + NP]], 1) for k in ks])

    models = []                                                     # (tag, 라벨, 색, 예측 위치)
    exps = list(dict.fromkeys(m.split("/")[0] for m in a.models))
    for spec in a.models:
        exp, rung = spec.split("/")
        p = os.path.join(CKPT, exp, f"{rung}_s{a.seed}.pt")
        s = json.load(open(p[:-3] + ".json"))["spec"]
        cfg = configs.golf(s["lon"], s["road"] == "bank+grade")
        assert s["lon"] == "force", "영상은 force 입력 모델만 (페달 입력은 따로 불러야 함)"
        su = np.concatenate([sg["s"], u_of(sg, s["road"])], 1)
        hist = torch.tensor(np.stack([su[k - H + 1:k + 1] for k in ks]), dtype=torch.float32)
        U = torch.tensor(np.stack([su[k + 1:k + NP, 3:] for k in ks]), dtype=torch.float32)
        pred = ladder.rollout(ladder.load_model(p, cfg), hist, U)     # (B, NP, 3) = k+1 .. k+NP 의 상태
        pose = torch.tensor(pose0, dtype=torch.float32)
        P = []
        for k in range(NP):                                         # 팀원 원본과 같은 위치 적분
            st = pred[:, k]
            pm = pose[:, 2] + 0.5 * DT * st[:, 2]
            pose = torch.stack([pose[:, 0] + DT * (st[:, 0] * torch.cos(pm) - st[:, 1] * torch.sin(pm)),
                                pose[:, 1] + DT * (st[:, 0] * torch.sin(pm) + st[:, 1] * torch.cos(pm)),
                                pose[:, 2] + DT * st[:, 2]], 1)
            P.append(pose[:, :2].numpy())
        lab = f"{KO[rung]}({rung})" + (f" {EXP_KO.get(exp, exp)}" if len(exps) > 1 else "")
        col = (COL if exp == exps[0] else COL2)[rung]
        models.append((spec, lab, col, np.nan_to_num(np.stack(P, 1), nan=1e4)))
    miss = {t: np.hypot(*(P[:, -1] - truth[:, -1]).T) for t, _, _, P in models}

    # HUD 재료: 가속도(g, 측정값), 타이어 한계 사용률(기준 R1 시드 0 이 학습한 타이어 곡선 기준)
    vx, vy, r = sg["s"].T
    ax_g, ay_g = sg["ax"] / 9.81, sg["ay"] / 9.81
    lon, delta = sg["lon"], sg["delta"]
    cfgb = configs.golf("force", True)
    Pw = {k: float(v) for k, v in zip(cfgb.pnames, ladder.load_model(os.path.join(CKPT, "base", "R1_s0.pt"), cfgb).g.values())}
    af = delta - np.arctan2(cfgb.lf * r + vy, np.maximum(vx, 1))
    ar = np.arctan2(cfgb.lr * r - vy, np.maximum(vx, 1))
    grid = torch.linspace(0.001, 0.4, 2000)
    apk_f = float(grid[torch.argmax(ladder.pacejka(grid, Pw["Bf"], Pw["Cf"], Pw["Df"], Pw["Ef"]))])
    apk_r = float(grid[torch.argmax(ladder.pacejka(grid, Pw["Br"], Pw["Cr"], Pw["Dr"], Pw["Er"]))])
    sat_f, sat_r = np.abs(af) / apk_f, np.abs(ar) / apk_r
    dmax = max(0.05, float(np.percentile(np.abs(delta[ks]), 99)))
    fmax = float(np.percentile(np.abs(lon[ks]), 99))

    track = Track()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    mv = MainView(fig, track, [0.0, 0.0, 0.62, 1.0])
    mm = MiniMap(fig, track, [0.845, 0.70, 0.145, 0.27])
    mdot = mm.add("#ffffff", 8)
    truth_ln, = mv.ax.plot([], [], color="white", lw=3.2, zorder=6, solid_capstyle="round")
    truth_end, = mv.ax.plot([], [], "o", color="white", ms=9, zorder=7)
    plines = [mv.ax.plot([], [], color=c, lw=2.4, zorder=5, solid_capstyle="round")[0] for _, _, c, _ in models]
    pends = [mv.ax.plot([], [], "o", color=c, ms=8, mec="k", mew=0.6, zorder=5)[0] for _, _, c, _ in models]
    car = Polygon(car_poly(0, 0, math.pi / 2), closed=True, fc="#f2f2f2", ec="k", lw=1, zorder=8)
    mv.ax.add_patch(car)
    trail, = mv.ax.plot([], [], color="white", lw=1, alpha=0.35, zorder=4)

    title = "실제 주행 기록 재생 · VW Golf, Thunderhill West"
    fig.text(0.015, 0.965, title, color=FG, fontsize=15, weight="bold", va="top")
    sub = f"차 앞의 선 = 실제 조작·노면을 넣었을 때 모델이 예측한 {a.horizon:g}초 동안의 길"
    if len(exps) == 1:
        sub += f" · 학습: {EXP_KO.get(exps[0], exps[0])}"
    fig.text(0.015, 0.92, sub, color=DIM, fontsize=11, va="top")
    lx = 0.015
    for lab, col in [("실제", "white")] + [(l, c) for _, l, c, _ in models]:
        fig.text(lx, 0.03, "━ " + lab, color=col, fontsize=11 if len(models) <= 4 else 9.5, weight="bold")
        lx += 0.05 + 0.0095 * len(lab)

    tx = 0.645
    sp_txt = fig.text(tx, 0.955, "", color=FG, fontsize=30, weight="bold", va="top")
    sp_sub = fig.text(tx, 0.87, "", color=DIM, fontsize=10, va="top")
    rd_txt = fig.text(tx, 0.84, "", color=DIM, fontsize=10, va="top")
    t_txt = fig.text(tx, 0.81, "", color=DIM, fontsize=9, va="top")
    axb = fig.add_axes([tx, 0.63, 0.14, 0.12]); style_ax(axb)
    axb.set_xlim(-1, 1); axb.set_ylim(-0.5, 1.6)
    axb.add_patch(plt.Rectangle((-1, 0.95), 2, 0.4, color="#2a2d33"))
    axb.add_patch(plt.Rectangle((-1, -0.25), 2, 0.4, color="#2a2d33"))
    steer_bar = axb.add_patch(plt.Rectangle((0, 0.95), 0, 0.4, color="#cfd3da"))
    pedal_bar = axb.add_patch(plt.Rectangle((0, -0.25), 0, 0.4, color="#2FBF71"))
    axb.axvline(0, color=DIM, lw=0.8)
    axb.text(-1, 1.42, "조향 (좌 ← → 우)", color=DIM, fontsize=9)
    axb.text(-1, 0.22, "제동 ← 앞바퀴 힘 → 구동", color=DIM, fontsize=9)
    axf = fig.add_axes([0.645, 0.30, 0.17, 0.33]); style_ax(axf)
    axf.set_aspect("equal")
    gmax = 1.2
    for rr in (0.5, 1.0):
        axf.add_patch(Circle((0, 0), rr, fill=False, ec="#4a4f58", lw=1))
        axf.text(0.03, rr + 0.03, f"{rr:g}g", color=DIM, fontsize=8)
    axf.axhline(0, color="#30343a", lw=0.8); axf.axvline(0, color="#30343a", lw=0.8)
    axf.set_xlim(-gmax * 1.12, gmax * 1.12); axf.set_ylim(-gmax * 1.12, gmax * 1.12)
    axf.set_title("가속도 (마찰원)", color=FG, fontsize=11)
    g_tail, = axf.plot([], [], color="#F5C542", lw=1.2, alpha=0.5)
    g_dot, = axf.plot([], [], "o", color="#F5C542", ms=9, mec="k")
    axt = fig.add_axes([0.85, 0.34, 0.12, 0.25]); style_ax(axt)
    axt.set_xlim(0, 1.5); axt.set_ylim(-0.6, 1.6)
    tb = [axt.barh(y, 0, height=0.5, color="#2FBF71")[0] for y in (1, 0)]
    for y, lab in ((1, "앞 타이어"), (0, "뒤 타이어")):
        axt.add_patch(plt.Rectangle((0, y - 0.25), 1.5, 0.5, fill=False, ec="#4a4f58"))
        axt.text(0, y + 0.33, lab, color=DIM, fontsize=9)
    axt.axvline(1.0, color="#E05A5A", lw=1, ls="--")
    axt.set_title("타이어 한계 사용률", color=FG, fontsize=11)
    axt.text(0, -0.55, "1 = 최대 힘 (R1 학습 타이어 기준)", color=DIM, fontsize=7.5)
    axe = fig.add_axes([0.70, 0.045, 0.27, 0.20]); style_ax(axe)
    ebars = axe.barh(range(len(models)), [0] * len(models), color=[c for _, _, c, _ in models], height=0.6)
    axe.set_yticks(range(len(models)))
    axe.set_yticklabels([l.split("(")[0] + (" " + l.split(" ")[-1] if len(exps) > 1 else "") for _, l, _, _ in models],
                        color=FG, fontsize=9.5)
    axe.tick_params(length=0)
    axe.invert_yaxis()
    emax = 3.0 * a.horizon
    axe.set_xlim(0, emax)
    axe.set_title(f"{a.horizon:g}초 뒤 빗나간 거리 (m) · 지금 / 누적 평균", color=FG, fontsize=11, loc="left")
    etxt = [axe.text(0, i, "", color=FG, va="center", fontsize=9) for i in range(len(models))]

    # 1초 뒤 끝점 확대창 (예측 차이가 수십 cm 라 큰 화면에서는 선이 겹친다)
    axz = fig.add_axes([0.02, 0.50, 0.19, 0.34]); style_ax(axz)
    axz.set_facecolor("#1d2025")
    axz.set_aspect("equal")
    ZR = 1.0
    axz.set_xlim(-ZR, ZR); axz.set_ylim(-ZR, ZR)
    for rr in (0.25, 0.5, 1.0):
        axz.add_patch(Circle((0, 0), rr, fill=False, ec="#3a3e45", lw=0.8))
        axz.text(rr * 0.71 + 0.02, -rr * 0.71 - 0.02, f"{rr:g} m", color=DIM, fontsize=7.5, va="top")
    axz.set_title(f"{a.horizon:g}초 뒤 끝점 확대 (흰 점 = 실제)", color=FG, fontsize=10.5)
    for sp in axz.spines.values():
        sp.set_visible(True); sp.set_color("#4a4f58")
    z_tl, = axz.plot([], [], color="white", lw=2.5, zorder=3)
    z_pl = [axz.plot([], [], color=c, lw=1.8, zorder=4)[0] for _, _, c, _ in models]
    z_pe = [axz.plot([], [], "o", color=c, ms=9, mec="k", mew=0.6, zorder=5)[0] for _, _, c, _ in models]
    z_truth, = axz.plot([], [], "o", color="white", ms=11, mec="k", zorder=6)

    def upd(i):
        k = ks[i]
        p0, psi0 = pose0[i, :2], pose0[i, 2]
        mv.frame(p0, psi0, vx[k])
        T = to_cam(truth[i], p0, psi0)
        truth_ln.set_data(np.r_[0, T[:, 0]], np.r_[0, T[:, 1]])
        truth_end.set_data([T[-1, 0]], [T[-1, 1]])
        for j, (tag, _, _, P) in enumerate(models):
            Q = to_cam(P[i], p0, psi0)
            plines[j].set_data(np.r_[0, Q[:, 0]], np.r_[0, Q[:, 1]])
            pends[j].set_data([Q[-1, 0]], [Q[-1, 1]])
            e = miss[tag][i]
            ebars[j].set_width(min(e, emax))
            etxt[j].set_text(f" {e:4.2f} / {miss[tag][:i + 1].mean():4.2f}")
            etxt[j].set_x(min(e, emax))
        tr = to_cam(np.stack([sg["x"][max(0, k - 75):k + 1], sg["y"][max(0, k - 75):k + 1]], 1), p0, psi0)
        trail.set_data(tr[:, 0], tr[:, 1])
        mdot.set_data([p0[0]], [p0[1]])
        sp_txt.set_text(f"{vx[k] * 3.6:5.0f} km/h")
        sp_sub.set_text(f"조향 {math.degrees(delta[k]):+5.1f}° · 회전 {math.degrees(r[k]):+5.1f}°/s")
        rd_txt.set_text(f"경사 {math.tan(sg['grade'][k]) * 100:+4.1f}% · 좌우 기울기 {math.degrees(sg['bank'][k]):+4.1f}°")
        t_txt.set_text(f"{RUN_KO.get(a.run, a.run)} (학습에 안 씀) · t = {(k - k0) * DT:5.1f} s")
        zt = T[-1]                                                  # 확대창: 실제 끝점 중심
        z_truth.set_data([0], [0])
        z_tl.set_data(T[:, 0] - zt[0], T[:, 1] - zt[1])
        for j, (_, _, _, P) in enumerate(models):
            Q = to_cam(P[i], p0, psi0) - zt
            z_pl[j].set_data(Q[:, 0], Q[:, 1])
            z_pe[j].set_data([Q[-1, 0]], [Q[-1, 1]])
        steer_bar.set_width(-np.clip(delta[k] / dmax, -1, 1))
        pv = np.clip(lon[k] / fmax, -1, 1)
        pedal_bar.set_width(pv)
        pedal_bar.set_color("#2FBF71" if pv >= 0 else "#E05A5A")
        lo = max(0, k - 25)
        g_tail.set_data(-ay_g[lo:k + 1], ax_g[lo:k + 1])
        g_dot.set_data([-ay_g[k]], [ax_g[k]])
        for b, sv in zip(tb, (sat_f[k], sat_r[k])):
            b.set_width(min(sv, 1.5))
            b.set_color("#E05A5A" if sv > 0.9 else ("#F5C542" if sv > 0.6 else "#2FBF71"))
        return []

    out = a.out or os.path.join(ROOT, "viz", "out", "ladder", f"predict_{'_'.join(exps)}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if a.preview is not None:
        upd(a.preview)
        fig.savefig(out.replace(".mp4", f"_f{a.preview}.png"), facecolor=BG)
        print("saved", out.replace(".mp4", f"_f{a.preview}.png"))
        return
    frames = range(0, len(ks), a.every)
    anim = animation.FuncAnimation(fig, upd, frames=frames, blit=False)
    anim.save(out, writer=animation.FFMpegWriter(fps=25 * a.speed / a.every, bitrate=3500))
    print("saved", out, len(frames), "frames; mean miss at horizon (m):",
          {t: round(float(v.mean()), 2) for t, v in miss.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2023-11_3", help="D7 시험 주행: 2023-10_4, 2023-11_3, 2023-11_8")
    ap.add_argument("--models", nargs="+", default=["base/R1", "base/R2", "base/R4", "base/R5"])
    ap.add_argument("--horizon", type=float, default=1.0, help="예측 길이 s (팀 계획 길이 1초. 팀원 원본 영상은 2초)")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=0.0, help="0 이면 구간 끝까지")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", type=int, default=None)
    main(ap.parse_args())
