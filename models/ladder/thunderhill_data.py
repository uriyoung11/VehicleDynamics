"""썬더힐 Stanford VW GTI 데이터 → 사다리 입력. 예시 로더 (데이터마다 이 부분만 새로 씀).

데이터: https://purl.stanford.edu/zb950hd3384 의 scirob_submission.zip
  Model_Learning/data/exp/exp_data/dry/exp_{2,4,5,6,8,9,10,11,12,15,16}.npz
  키: ux, uy, r, delta, fxf (100 Hz). 위치(x, y)는 없음 → 트랙 모양은 이 데이터로 못 만듦
  dry.npz(합본)는 쓰지 말 것: 4스텝 창을 이미 잘라 섞은 파일이라 주행 단위로 나눌 수 없음

주행 성격 (파일 값으로 확인, 무엇을 달린 건지는 [추정])
  exp_2, 8, 9, 10, 16 : vx 최대 약 33 m/s, |조향| 최대 0.15~0.16 rad → 코스 주행으로 보임
  exp_4, 5, 6, 11, 12, 15 : vx 최대 16~19 m/s, |조향| 최대 0.26~0.47 rad → 스키드패드 오벌로 보임

정해야 할 것 (팀 결정, 아래 값은 예시)
  - dt: 100 Hz 그대로(0.01 s) 쓸지, 4개마다 뽑아 0.04 s(다른 데이터·MPPI 와 같은 간격)로 쓸지
  - 학습·검증·시험을 어떤 주행으로 나눌지 (반드시 주행 단위)
  - 저속 구간 기준 (다른 데이터와 같이 vx > 5 m/s 만 씀)
"""
import glob
import os

import numpy as np

EXAMPLE_SPLIT = dict(  # 예시. 두 성격의 주행이 모든 쪽에 들어가게 나눈 것일 뿐, 정해진 분할 아님
    train=["exp_2", "exp_8", "exp_10", "exp_4", "exp_6", "exp_11", "exp_15"],
    val=["exp_16", "exp_5"],
    test=["exp_9", "exp_12"],
)


def load_run(path, every=4, vmin=5.0, min_len=40):
    """npz 하나 → 끊김 없는 구간 [(s (T,3), u (T,3)), ...]. every=4 이면 100 Hz → 25 Hz (dt 0.04).
    u = [fxf (N), delta (rad), bank=0]."""
    d = np.load(path, allow_pickle=False)
    s = np.stack([d["ux"], d["uy"], d["r"]], 1).astype(np.float64)
    u = np.stack([d["fxf"], d["delta"], np.zeros_like(d["ux"])], 1).astype(np.float64)
    s, u = s[::every], u[::every]  # 단순 솎아내기. 원 데이터에 저역 통과가 이미 돼 있는지는 확인 필요
    ok = s[:, 0] > vmin
    segs, i, n = [], 0, len(s)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            segs.append((s[i:j + 1], u[i:j + 1]))
        i = j + 1
    return segs


def load_split(data_dir, names, **kw):
    out = []
    for nm in names:
        out += load_run(os.path.join(data_dir, nm + ".npz"), **kw)
    return out


def available(data_dir):
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(data_dir, "exp_*.npz")))
