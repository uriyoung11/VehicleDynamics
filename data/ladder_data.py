"""팀원 사다리(R1~R6, models/ladder) 형식의 데이터: D7(Golf, 주 데이터)과 D1(GTI, 함께 학습 비교용).

  from data.ladder_data import d7_segments, d1_segments
  segs = d7_segments("train", lon="force", road="bank+grade")      # [(s (T,3), u (T,nu)), ...]

형식 (models/ladder/README_team.md 의 약속)
  s = [vx, vy, r],  u = [lon, delta, bank(, grade)],  dt 0.04 s (D7 200 Hz는 8개, D1 100 Hz는 4개씩 평균)
  vx > 5 m/s 연속 구간, 길이 38 스텝 이상. 학습·검증·시험은 주행(파일) 단위: data.datasets.D7_SPLIT, D1_SPLIT

lon (종방향 입력)
  "force": 앞바퀴 종방향 힘 Fxf [N] = 구동(엔진 토크 × GearRatio × 보정계수, 전륜) − 앞 브레이크 힘.
           측정 가속도를 쓰지 않아 정답이 새지 않는다. 보정계수: data.datasets.D7_K_DRIVE/D7_K_BRAKE (R² 0.937)
  "pedal": 스로틀/100, 제동 중이면 −브레이크 압력/50 bar (팀원·Deep Dynamics 형식)
road (노면)
  "none": bank = 0
  "bank": 트랙 파일의 좌우 기울기(rad). 팀원 방식(모델 안에서 kb·g·sin(bank)으로 쓰임, kb 학습)
  "bank+grade": 위 + 경사(rad, 오르막 +) 채널. 우리 방식(dvx −= g·sin(경사)). 설정에 use_grade=True 필요
  트랙 파일: data/tracks/ThunderhillWest_centerline.csv (python -m data.thunderhill_west), 위치 [posE, posN]로 찾는다.

D1(GTI)은 Golf 설정으로 함께 학습하려고 힘을 "같은 가속도"가 나오게 환산한다:
  구동 lon = fxf·m_Golf/m_GTI,  제동 lon = fxf·(0.75·m_Golf)/(0.66·m_GTI)  (Golf 설정의 제동 앞 몫 0.75, GTI 0.66)
  D1에는 위치·페달이 없어 노면은 0, lon은 force만 된다.
"""
import glob, os
import numpy as np
import pandas as pd

from data.datasets import D7_DIR, D7_DUP, D7_SPLIT, D7_K_DRIVE, D7_K_BRAKE, D1_DIR, D1_SPLIT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = os.path.join(ROOT, "data", "tracks", "ThunderhillWest_centerline.csv")
M_GOLF, M_GTI, BF_GOLF, BF_GTI = 1630.0, 1776.2, 0.75, 0.66
HIST, MIN_LEN, VMIN = 8, 38, 5.0


def _blocks(a, k):
    n = len(a) // k * k
    return a[:n].reshape(n // k, k, *a.shape[1:]).mean(1)


def _segments(s, u, vmin=VMIN, min_len=MIN_LEN):
    ok = s[:, 0] > vmin
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [(s[i:j], u[i:j]) for i, j in zip(edges[::2], edges[1::2]) if j - i >= min_len]


def _road_lookup():
    from scipy.spatial import cKDTree
    T = np.loadtxt(TRACK, delimiter=",", comments="#")
    assert T.shape[1] >= 7, "트랙 파일에 경사·기울기 열이 없다. python -m data.thunderhill_west 를 먼저 실행"
    tree = cKDTree(T[:, :2])
    return lambda en: T[tree.query(en)[1]][:, [6, 5]]          # [bank, grade]


def d7_segments(split, lon="force", road="none"):
    assert lon in ("force", "pedal") and road in ("none", "bank", "bank+grade")
    look = _road_lookup() if road != "none" else None
    cols = ["Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad", "engineTorque_Nm", "GearRatio",
            "brake_fl_N", "brake_fr_N", "throttleCmd_percent", "brake_fl_bar", "posE_m", "posN_m"]
    out = []
    for f in sorted(glob.glob(os.path.join(D7_DIR, "*.csv"))):
        name = os.path.basename(f)[:-4]
        if name in D7_DUP:
            continue
        sp = next((k for k, runs in D7_SPLIT.items() if name in runs), "train")
        if split not in ("all", sp):
            continue
        a = _blocks(pd.read_csv(f, comment="#", usecols=cols)[cols].to_numpy(np.float64), 8)   # 200 Hz → 25 Hz
        vx, vy, r, delta = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        if lon == "force":
            lo = D7_K_DRIVE / D7_K_BRAKE * a[:, 4] * a[:, 5] - (a[:, 6] + a[:, 7])
        else:
            lo = np.where(a[:, 9] > 0.5, -a[:, 9] / 50.0, a[:, 8] / 100.0)
        road_cols = look(a[:, 10:12]) if look else np.zeros((len(a), 2))
        u = np.c_[lo, delta, road_cols[:, 0]] if road != "bank+grade" else np.c_[lo, delta, road_cols]
        if road == "none":
            u[:, 2] = 0.0
        out += _segments(np.c_[vx, vy, r], u)
    return out


def d1_segments(split, road="none"):
    """D1(GTI, Spielberg 2019 실험 데이터)을 Golf 설정에 맞춰 환산. force 전용, 노면 0."""
    out = []
    for f in sorted(glob.glob(os.path.join(D1_DIR, "exp_*.npz"))):
        name = os.path.basename(f)[:-4]
        sp = next((k for k, runs in D1_SPLIT.items() if name in runs), "train")
        if split not in ("all", sp):
            continue
        z = np.load(f)
        a = _blocks(np.stack([z["ux"], z["uy"], z["r"], z["delta"], z["fxf"]], 1).astype(np.float64), 4)   # 100 Hz → 25 Hz
        fx = a[:, 4]
        lo = np.where(fx >= 0, fx * M_GOLF / M_GTI, fx * BF_GOLF * M_GOLF / (BF_GTI * M_GTI))
        nroad = 2 if road == "bank+grade" else 1
        u = np.c_[lo, a[:, 3], np.zeros((len(a), nroad))]
        out += _segments(a[:, :3], u)
    return out


def summary(segs, dt=0.04):
    n = sum(len(s) for s, _ in segs)
    return dict(segments=len(segs), minutes=round(n * dt / 60, 1))
