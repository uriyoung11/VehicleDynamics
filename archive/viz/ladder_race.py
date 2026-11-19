"""팀원 방식(iac_viz2d.py race 모드) 고스트 레이스 영상: 같은 세계(Golf 시뮬레이터)·같은 MPPI 로
사다리 모델들이 Thunderhill West 를 동시에 달린 결과(control/ladder_race.py)를 한 화면에 재생한다.

  python viz/ladder_race.py                         # -> viz/out/ladder/race_base.mp4
  python viz/ladder_race.py --suffix __x1.06            # D7 속도 × 1.06 한계 시험 -> race_base__x1.06.mp4
  python viz/ladder_race.py --follow R2 --preview 500
"""
import argparse, json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from ladder_predict import BG, FG, DIM, MainView, MiniMap, Track, car_poly, to_cam   # noqa: E402

DT = 0.04
RES = os.path.join(ROOT, "eval", "results", "ladder_race")
RACE_COL = {"R1": "#4C8DF6", "R2": "#2FBF71", "R3": "#3CC6C6", "R4": "#B07CF2", "R5": "#F5883A", "R6": "#E05A5A", "SIM": "#f2f2f2"}
NAME = {"R1": "물리", "R2": "물리+보정", "R3": "신경망 타이어", "R4": "DD식", "R5": "신경망", "R6": "블랙박스", "SIM": "정답 모델"}
STAT = {"finished": "완주", "off_track": "트랙 이탈", "stalled": "멈춤", "world_nan": "세계 발산", "timeout": "시간 초과"}


def main(a):
    track = Track()
    ntr = int(np.load(os.path.join(ROOT, "data", "tracks", "thunderhill_race.npz"))["x"].shape[0])
    cars = []
    for r in ["R1", "R2", "R3", "R4", "R5", "R6"] + (["SIM"] if a.sim else []):
        tag = (f"{a.exp}__{r}_s{a.seed}" if r != "SIM" else "SIM") + a.suffix
        p = os.path.join(RES, tag + ".json")
        if not os.path.exists(p):
            continue
        L = np.load(os.path.join(RES, tag + "_traj.npz"))
        res = json.load(open(p))
        di = np.diff(L["idx"].astype(int))
        di = np.where(di < -ntr / 2, di + ntr, di)
        cars.append(dict(tag=r, name=NAME[r], X=L["X"], Y=L["Y"], psi=L["psi"], vx=L["vx"], prog=np.r_[0, np.cumsum(di)],
                         status=res["status"], lap=res["lap_time"], col=RACE_COL[r]))
    T = max(len(c["X"]) for c in cars)
    if a.dur:
        T = min(T, int(a.dur / DT))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    mv = MainView(fig, track, [0.0, 0.0, 0.62, 1.0], zoom=(10.0, 0.55))
    mm = MiniMap(fig, track, [0.70, 0.62, 0.28, 0.33])
    for c in cars:
        c["patch"] = Polygon(car_poly(0, 0, 0), closed=True, fc=c["col"], ec="k", lw=1, zorder=8,
                             alpha=0.55 if c["tag"] == "SIM" else 1.0)
        mv.ax.add_patch(c["patch"])
        c["trail"], = mv.ax.plot([], [], color=c["col"], lw=1.6, alpha=0.6, zorder=5)
        c["lab"] = mv.ax.text(0, 0, c["tag"], color=c["col"], fontsize=10, weight="bold", zorder=9, ha="left", va="bottom")
        c["mark"], = mv.ax.plot([], [], "x", color=c["col"], ms=16, mew=3.5, zorder=9)
        c["mdot"] = mm.add(c["col"], 8)
    fig.text(0.015, 0.965, "Thunderhill West · Golf 시뮬레이터 세계에서 제어 주행", color=FG, fontsize=15, weight="bold", va="top")
    sc = json.load(open(os.path.join(RES, "SIM" + a.suffix + ".json"))).get("scale", 1.0) if a.sim else 1.0
    if sc != 1.0:
        fig.text(0.015, 0.875, f"목표 속도 = D7 운전자 기록 × {sc:g}  (기록은 이미 코너 횡 1.07 g · 반투명 흰 차 = 정답 모델)",
                 color="#F5C542", fontsize=11, va="top")
    fig.text(0.015, 0.92, "같은 제어기(MPPI, 1초 계획), 머릿속 차 모델만 다름 · 학습: D7, 힘 입력, 기울기+경사",
             color=DIM, fontsize=11, va="top")
    t_txt = fig.text(0.015, 0.03, "", color=DIM, fontsize=12)
    rows = [fig.text(0.70, 0.54 - 0.07 * j, "", fontsize=12.5, va="top") for j in range(len(cars))]
    fig.text(0.70, 0.585, "순위 · 모델 · 속도 · 상태", color=DIM, fontsize=10)
    cam = {}

    def upd(fi):
        js = [min(fi, len(c["X"]) - 1) for c in cars]
        prog = [c["prog"][j] for c, j in zip(cars, js)]
        alive = [p if (fi < len(c["X"]) or c["status"] == "finished") and c["tag"] != "SIM" else -1 for c, p in zip(cars, prog)]
        lead = int(np.argmax(alive))
        if max(alive) < 0 and any(c["tag"] == "SIM" for c in cars):        # 모델 차가 모두 멈추면 정답 모델을 따라간다
            lead = [c["tag"] for c in cars].index("SIM")
        if a.follow:
            lead = [c["tag"] for c in cars].index(a.follow)
        cl, jl = cars[lead], js[lead]
        tp, tpsi = np.array([cl["X"][jl], cl["Y"][jl]]), cl["psi"][jl]
        if not cam:
            cam.update(p=tp, psi=tpsi)
        else:
            cam["p"] = cam["p"] + 0.2 * (tp - cam["p"])
            cam["psi"] = cam["psi"] + 0.2 * ((tpsi - cam["psi"] + math.pi) % (2 * math.pi) - math.pi)
        p0, psi0 = cam["p"], cam["psi"]
        mv.frame(p0, psi0, cl["vx"][jl])
        for c, j in zip(cars, js):
            c["patch"].set_xy(to_cam(car_poly(c["X"][j], c["Y"][j], c["psi"][j]), p0, psi0))
            tr = to_cam(np.stack([c["X"][max(0, j - 50):j + 1], c["Y"][max(0, j - 50):j + 1]], 1), p0, psi0)
            c["trail"].set_data(tr[:, 0], tr[:, 1])
            c["q"] = to_cam(np.array([c["X"][j], c["Y"][j]]), p0, psi0)
            ended = fi >= len(c["X"]) - 1
            c["mark"].set_data(*(([c["q"][0]], [c["q"][1]]) if ended and c["status"] != "finished" else ([], [])))
            c["mdot"].set_data([c["X"][j]], [c["Y"][j]])
        ys = sorted(range(len(cars)), key=lambda i: cars[i]["q"][1])
        gap, last = mv.halfw * 0.06, -1e9
        for i in ys:
            y = max(cars[i]["q"][1] + 1.0, last + gap)
            cars[i]["lab"].set_position((cars[i]["q"][0] + 2.0, y))
            last = y

        def rank_key(i):
            c = cars[i]
            if c["status"] == "finished" and fi >= len(c["X"]) - 1:
                return (0, c["lap"])
            return (1, -prog[i])
        for rk, ci in enumerate(sorted(range(len(cars)), key=rank_key)):
            c, j = cars[ci], js[ci]
            ended = fi >= len(c["X"]) - 1
            st = (STAT[c["status"]] + (f" {c['lap']:.1f}s" if c["lap"] else "")) if ended else f"{c['prog'][j] / ntr * 100:4.0f}%"
            rows[rk].set_text(f"{rk + 1}.  {c['tag']} {c['name']:<6}  {c['vx'][j] * 3.6:4.0f} km/h   {st}")
            rows[rk].set_color(c["col"] if not (ended and c["status"] != "finished") else "#777")
        t_txt.set_text(f"t = {fi * DT:5.1f} s   (카메라: {cl['tag']} {cl['name']})")
        return []

    out = a.out or os.path.join(ROOT, "viz", "out", "ladder", f"race_{a.exp}{a.suffix}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if a.preview is not None:
        upd(a.preview)
        fig.savefig(out.replace(".mp4", f"_f{a.preview}.png"), facecolor=BG)
        print("saved", out.replace(".mp4", f"_f{a.preview}.png"))
        return
    frames = list(range(0, T, a.every)) + [T - 1] * 25
    anim = animation.FuncAnimation(fig, upd, frames=frames, blit=False)
    anim.save(out, writer=animation.FFMpegWriter(fps=25 * a.speed / a.every, bitrate=3500))
    print("saved", out, len(frames), "frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="base")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--sim", type=int, default=1, help="정답 모델(SIM) 차를 반투명으로 같이 그린다")
    ap.add_argument("--follow", default=None)
    ap.add_argument("--suffix", default="", help="결과 파일 꼬리 (예: __x1.06 = D7 속도 × 1.06)")
    ap.add_argument("--dur", type=float, default=0)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", type=int, default=None)
    main(ap.parse_args())
