"""시각화 도구 시험 2: Foxglove. 공용 장면(viz/out/replay.npz)을 MCAP 파일로 기록한다.

  python -m viz.tools.foxglove_demo        # -> viz/out/tools/foxglove.mcap, foxglove_layout.json

열기: Foxglove 앱(https://app.foxglove.dev 또는 데스크톱 앱, 무료 계정)에서 foxglove.mcap 열기
      → 왼쪽 위 Layout 메뉴 → Import from file → foxglove_layout.json
      로그인 없이 쓰려면 오픈소스 포크 Lichtblick(https://github.com/lichtblick-suite/lichtblick)도 같은 파일을 연다.

토픽
  /tf        world→car, car→바퀴(앞바퀴 조향)
  /track     트랙(아스팔트, 연석, 젖은 구간) — 처음 한 번
  /car       차체·바퀴 (car 좌표계에 붙어 움직임)
  /forces    타이어 횡력 화살표(색: 초록 평소 / 노랑 한계 진입 / 빨강 포화)
  /pred      1초 예측 궤적 + 1초 뒤 고스트 차 (흰색 실제, 주황 M0, 파랑 B1)
  /trail     지나온 길(색: 가장자리까지 여유)
  /state     그래프용 수치(JSON): 속도, μ, 여유, 타이어 비율, 모델별 1초 위치 오차·예측 최소 여유
"""
import json, math, os
import numpy as np
import foxglove
import foxglove.messages as M
from foxglove.channels import SceneUpdateChannel, FrameTransformsChannel

from viz.replay import load, OUT as REPLAY
from viz.demo_rerun import Margin  # noqa: F401  (여유 계산은 replay에 이미 들어 있음)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "viz", "out", "tools")
COL = {"truth": (1, 1, 1, 0.6), "m0": (1.0, 0.54, 0.24, 0.8), "b1": (0.31, 0.64, 1.0, 0.8)}
OK, WARN, BAD = (0.27, 0.78, 0.43, 1), (0.98, 0.78, 0.16, 1), (0.94, 0.27, 0.24, 1)


def c(rgba):
    return M.Color(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])


def q_yaw(yaw):
    return M.Quaternion(x=0, y=0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def q_roll90():
    return M.Quaternion(x=math.sin(math.pi / 4), y=0, z=0, w=math.cos(math.pi / 4))


def pose(x=0, y=0, z=0, q=None):
    return M.Pose(position=M.Vector3(x=x, y=y, z=z), orientation=q or M.Quaternion(x=0, y=0, z=0, w=1))


def P3(x, y, z=0.0):
    return M.Point3(x=float(x), y=float(y), z=float(z))


def margin_col(m):
    return BAD if m < 0 else WARN if m < 0.25 else OK


def ratio_col(r):
    return BAD if r >= 1 else WARN if r >= 0.5 else OK


def track_entity(R, ts):
    L, Rt, C = R["track_left"], R["track_right"], R["track_center"]
    n = len(L)
    pts, idx = [], []
    for i in range(n):
        j = (i + 1) % n
        pts += [P3(*L[i]), P3(*Rt[i]), P3(*L[j]), P3(*L[j]), P3(*Rt[i]), P3(*Rt[j])]
    tris = [M.TriangleListPrimitive(pose=pose(), points=pts, color=c((0.24, 0.25, 0.28, 1)))]
    wi = R["wet_idx"]
    if len(wi) > 1:
        wp = []
        for i, j in zip(wi[:-1], wi[1:]):
            wp += [P3(*L[i], 0.003), P3(*Rt[i], 0.003), P3(*L[j], 0.003), P3(*L[j], 0.003), P3(*Rt[i], 0.003), P3(*Rt[j], 0.003)]
        tris.append(M.TriangleListPrimitive(pose=pose(), points=wp, color=c((0.18, 0.5, 0.97, 0.55))))
    lines = []
    for line in (L, Rt):
        closed = np.r_[line, line[:1]]
        cols = [c((0.88, 0.16, 0.16, 1) if (k // 3) % 2 else (0.95, 0.95, 0.95, 1)) for k in range(len(closed))]
        lines.append(M.LinePrimitive(type=M.LinePrimitiveLineType.LineStrip, pose=pose(), thickness=0.09, scale_invariant=False,
                                     points=[P3(x, y, 0.01) for x, y in closed], colors=cols))
    lines.append(M.LinePrimitive(type=M.LinePrimitiveLineType.LineList, pose=pose(), thickness=0.02, scale_invariant=False,
                                 points=[P3(x, y, 0.008) for x, y in C[: len(C) // 2 * 2]], color=c((0.8, 0.8, 0.8, 0.5))))
    return M.SceneEntity(timestamp=ts, frame_id="world", id="track", triangles=tris, lines=lines)


def car_entity(R, ts, frame="car", eid="car", body=(1, 0.77, 0, 1), ghost=False):
    a, b = float(R["car_a"]), float(R["car_b"])
    cx = (a - b) / 2
    cubes = [M.CubePrimitive(pose=pose(cx, 0, 0.06), size=M.Vector3(x=a + b + 0.18, y=0.23, z=0.06), color=c(body))]
    if not ghost:
        cubes += [M.CubePrimitive(pose=pose(cx - 0.02, 0, 0.11), size=M.Vector3(x=0.26, y=0.18, z=0.05), color=c((0.12, 0.12, 0.13, 1))),
                  M.CubePrimitive(pose=pose(a + 0.11, 0, 0.05), size=M.Vector3(x=0.04, y=0.24, z=0.024), color=c((0.86, 0.16, 0.16, 1)))]
    cyl = []
    if not ghost:
        for x, y in ((-b, 0.14), (-b, -0.14)):
            cyl.append(M.CylinderPrimitive(pose=pose(x, y, 0.05, q_roll90()), size=M.Vector3(x=0.1, y=0.1, z=0.045),
                                           bottom_scale=1, top_scale=1, color=c((0.08, 0.08, 0.08, 1))))
    return M.SceneEntity(timestamp=ts, frame_id=frame, id=eid, frame_locked=True, cubes=cubes, cylinders=cyl)


def wheel_entity(ts, frame):
    return M.SceneEntity(timestamp=ts, frame_id=frame, id=frame, frame_locked=True, cylinders=[
        M.CylinderPrimitive(pose=pose(0, 0, 0, q_roll90()), size=M.Vector3(x=0.1, y=0.1, z=0.045), bottom_scale=1, top_scale=1,
                            color=c((0.08, 0.08, 0.08, 1)))])


def main():
    R = load(REPLAY)
    models = [k[5:] for k in R if k.startswith("path_")]
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "foxglove.mcap")
    if os.path.exists(path):
        os.remove(path)
    writer = foxglove.open_mcap(path)
    tf_ch, track_ch, car_ch = FrameTransformsChannel("/tf"), SceneUpdateChannel("/track"), SceneUpdateChannel("/car")
    force_ch, pred_ch, trail_ch = SceneUpdateChannel("/forces"), SceneUpdateChannel("/pred"), SceneUpdateChannel("/trail")
    state_ch = foxglove.Channel("/state", schema={"type": "object"}, message_encoding="json")

    a, b = float(R["car_a"]), float(R["car_b"])
    frames = R["frames"]
    t_base = 1_700_000_000_000_000_000
    n = len(R["t"])
    fde = {k: np.linalg.norm(R["path_" + k][:, -1] - R["truth_path"][:, -1], axis=1) for k in models}
    from viz.demo_rerun import Margin as _M  # 예측 궤적 최소 여유 (replay에는 없어서 여기서 계산)
    from viz.demo_track import load_track
    mfun = _M(load_track("Spielberg"))
    mp = {k: mfun(R["truth_path"] if k == "truth" else R["path_" + k])[0].min(1) for k in ["truth", *models]}

    for s in range(frames[0], frames[-1] + 1):
        ln = t_base + int(R["t"][s] * 1e9)
        ts = M.Timestamp(sec=int(ln // 1_000_000_000), nsec=int(ln % 1_000_000_000))
        X, Y, psi, de = (float(R[k][s]) for k in ("X", "Y", "psi", "delta"))
        tfs = [M.FrameTransform(timestamp=ts, parent_frame_id="world", child_frame_id="car",
                                translation=M.Vector3(x=X, y=Y, z=0), rotation=q_yaw(psi))]
        for w, y in (("wheel_fl", 0.14), ("wheel_fr", -0.14)):
            tfs.append(M.FrameTransform(timestamp=ts, parent_frame_id="car", child_frame_id=w,
                                        translation=M.Vector3(x=a, y=y, z=0.05), rotation=q_yaw(de)))
        tf_ch.log(M.FrameTransforms(transforms=tfs), log_time=ln)
        if s == frames[0]:
            track_ch.log(M.SceneUpdate(entities=[track_entity(R, ts)]), log_time=ln)
            car_ch.log(M.SceneUpdate(entities=[car_entity(R, ts), wheel_entity(ts, "wheel_fl"), wheel_entity(ts, "wheel_fr")]), log_time=ln)
        # 타이어 횡력 화살표
        arrows = []
        for x, fy, rat, steer in ((a, R["fy_f"][s], R["ratio_f"][s], de), (-b, R["fy_r"][s], R["ratio_r"][s], 0.0)):
            L = abs(0.45 * fy)
            if L > 0.01:
                yaw = steer + (math.pi / 2 if fy > 0 else -math.pi / 2)
                arrows.append(M.ArrowPrimitive(pose=pose(x, 0, 0.17, q_yaw(yaw)), shaft_length=max(L - 0.06, 0.01), shaft_diameter=0.03,
                                               head_length=0.06, head_diameter=0.07, color=c(ratio_col(rat))))
        force_ch.log(M.SceneUpdate(entities=[M.SceneEntity(timestamp=ts, frame_id="car", id="forces", frame_locked=True, arrows=arrows)]),
                     log_time=ln)
        if s % 5 == 0:
            a0 = max(0, s - 300)
            pts = [P3(R["X"][i], R["Y"][i], 0.02) for i in range(a0, s + 1, 3)]
            cols = [c(margin_col(R["margin"][i])) for i in range(a0, s + 1, 3)]
            trail_ch.log(M.SceneUpdate(entities=[M.SceneEntity(timestamp=ts, frame_id="world", id="trail", lines=[
                M.LinePrimitive(type=M.LinePrimitiveLineType.LineStrip, pose=pose(), thickness=0.04, scale_invariant=False,
                                points=pts, colors=cols)])]), log_time=ln)
        fi = int(np.searchsorted(frames, s))
        if fi < len(frames) and frames[fi] == s:
            ents = []
            for k in ["truth", *models]:
                p = R["truth_path"][fi] if k == "truth" else R["path_" + k][fi]
                ents.append(M.SceneEntity(timestamp=ts, frame_id="world", id=f"path_{k}", lines=[
                    M.LinePrimitive(type=M.LinePrimitiveLineType.LineStrip, pose=pose(), thickness=0.05 if k == "truth" else 0.04,
                                    scale_invariant=False, points=[P3(x, y, 0.03) for x, y in p[::4]], color=c(COL[k][:3] + (1,)))]))
                d = p[-1] - p[-4]; hd = math.atan2(d[1], d[0])
                ents.append(M.SceneEntity(timestamp=ts, frame_id="world", id=f"ghost_{k}", cubes=[
                    M.CubePrimitive(pose=pose(p[-1, 0] + (a - b) / 2 * math.cos(hd), p[-1, 1] + (a - b) / 2 * math.sin(hd), 0.07, q_yaw(hd)),
                                    size=M.Vector3(x=a + b + 0.18, y=0.23, z=0.1), color=c(COL[k]))]))
            pred_ch.log(M.SceneUpdate(entities=ents), log_time=ln)
            st = {"speed": float(R["ux"][s]), "mu": float(R["mu"][s]), "margin": float(R["margin"][s]),
                  "ratio_front": float(R["ratio_f"][s]), "ratio_rear": float(R["ratio_r"][s]),
                  "margin_1s_min": {k: float(mp[k][fi]) for k in mp},
                  "pos_error_1s": {k: float(fde[k][fi]) for k in models}}
            state_ch.log(json.dumps(st).encode(), log_time=ln)
    writer.close()
    print("saved", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")

    # 레이아웃: 왼쪽 3D(차 따라가기), 오른쪽 그래프 3개
    plot = lambda title, series: {"title": title, "paths": [
        {"value": v, "enabled": True, "timestampMethod": "receiveTime", "label": lab} for v, lab in series],
        "showLegend": True, "legendDisplay": "top", "isSynced": True}
    layout = {
        "configById": {
            "3D!main": {"followTf": "car", "followMode": "follow-pose",
                        "cameraState": {"perspective": True, "distance": 4.5, "phi": 62, "thetaOffset": 180,
                                        "targetOffset": [1.0, 0, 0], "target": [0, 0, 0], "targetOrientation": [0, 0, 0, 1], "fovy": 45,
                                        "near": 0.01, "far": 500},
                        "scene": {"backgroundColor": "#18281c", "enableStats": False},
                        "topics": {t: {"visible": True} for t in ("/track", "/car", "/forces", "/pred", "/trail")},
                        "layers": {"grid": {"visible": False, "layerId": "foxglove.Grid"}}},
            "3D!top": {"followTf": "car", "followMode": "follow-position",
                       "cameraState": {"perspective": False, "distance": 40, "phi": 0, "thetaOffset": 0, "targetOffset": [0, 0, 0],
                                       "target": [0, 0, 0], "targetOrientation": [0, 0, 0, 1], "fovy": 45, "near": 0.01, "far": 500},
                       "topics": {t: {"visible": True} for t in ("/track", "/car", "/pred", "/trail")},
                       "layers": {"grid": {"visible": False, "layerId": "foxglove.Grid"}}},
            "Plot!margin": plot("트랙 가장자리까지 여유 [m]", [("/state.margin", "지금"), ("/state.margin_1s_min.truth", "1초 안 최소 (실제)")]
                                + [(f"/state.margin_1s_min.{k}", f"1초 안 최소 ({k})") for k in models]),
            "Plot!err": plot("1초 뒤 위치 오차 [m]", [(f"/state.pos_error_1s.{k}", k) for k in models]),
            "Plot!state": plot("속도·μ·타이어", [("/state.speed", "속도"), ("/state.mu", "μ"), ("/state.ratio_front", "앞 |α|/α_peak"),
                                                 ("/state.ratio_rear", "뒤 |α|/α_peak")]),
        },
        "globalVariables": {}, "userNodes": {}, "playbackConfig": {"speed": 1},
        "layout": {"direction": "row", "splitPercentage": 62,
                   "first": {"direction": "column", "splitPercentage": 70, "first": "3D!main", "second": "3D!top"},
                   "second": {"direction": "column", "first": "Plot!margin",
                              "second": {"direction": "column", "first": "Plot!err", "second": "Plot!state"}}},
    }
    with open(os.path.join(OUTDIR, "foxglove_layout.json"), "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=1)
    print("saved", os.path.join(OUTDIR, "foxglove_layout.json"))


if __name__ == "__main__":
    main()
