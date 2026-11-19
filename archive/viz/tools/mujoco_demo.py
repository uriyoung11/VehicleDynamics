"""시각화 도구 시험 1: MuJoCo. 공용 장면(viz/out/replay.npz)을 MuJoCo 장면으로 다시 그려 영상으로 저장.

  python -m viz.tools.mujoco_demo            # -> viz/out/tools/mujoco.mp4, mujoco.png

MuJoCo 물리는 쓰지 않는다. 차 위치·자세·조향각을 매 프레임 직접 넣고(kinematic replay) 그리기만 한다.
"""
import argparse, math, os, time
import numpy as np
import mujoco
import imageio.v2 as imageio

from viz.replay import load, OUT as REPLAY

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "viz", "out", "tools")
RGBA = {"truth": (1, 1, 1, 0.55), "m0": (1.0, 0.54, 0.24, 0.75), "b1": (0.31, 0.64, 1.0, 0.75)}


def slab(left, right, z, th=0.004):
    """경계선 사이를 얇은 판 메쉬로 (MuJoCo는 부피가 있는 메쉬가 필요)."""
    n = len(left)
    top = np.vstack([np.c_[left, np.full(n, z)], np.c_[right, np.full(n, z)]])
    v = np.vstack([top, top - [0, 0, th]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, n + i, j], [j, n + i, n + j]]                       # 윗면
        f += [[2 * n + i, 2 * n + j, 3 * n + i], [2 * n + j, 3 * n + j, 3 * n + i]]   # 아랫면
    return v, np.array(f)


def car_xml(name, a, b, body_rgba, dark_rgba, wheel_rgba, ghost=False):
    tag = 'mocap="true"' if ghost else ""
    joint = "" if ghost else '<freejoint name="car_free"/>'
    cx = (a - b) / 2
    g = 'contype="0" conaffinity="0"'
    wheels = ""
    for wn, x, y in (("fl", a, 0.14), ("fr", a, -0.14), ("rl", -b, 0.14), ("rr", -b, -0.14)):
        steer = f'<joint name="{name}_steer_{wn}" type="hinge" axis="0 0 1"/>' if (wn[0] == "f" and not ghost) else ""
        wheels += (f'<body name="{name}_w{wn}" pos="{x} {y} 0.05">{steer}'
                   f'<geom type="cylinder" size="0.05 0.022" euler="90 0 0" rgba="{wheel_rgba}" {g} mass="0.01"/></body>')
    return (f'<body name="{name}" {tag}>{joint}'
            f'<geom type="box" pos="{cx} 0 0.06" size="{(a + b) / 2 + 0.09} 0.115 0.03" rgba="{body_rgba}" {g} mass="1"/>'
            f'<geom type="box" pos="{cx - 0.02} 0 0.11" size="0.13 0.09 0.025" rgba="{dark_rgba}" {g} mass="0.1"/>'
            f'<geom type="box" pos="{a + 0.11} 0 0.05" size="0.02 0.12 0.012" rgba="0.86 0.16 0.16 1" {g} mass="0.01"/>'
            + ("" if ghost else '<camera name="chase" pos="-2.0 0 0.95" xyaxes="0 -1 0 0.42 0 0.91" fovy="55"/>')
            + wheels + "</body>")


def build_model(R, models):
    L, Rt, C = R["track_left"], R["track_right"], R["track_center"]
    a, b = float(R["car_a"]), float(R["car_b"])
    v, f = slab(L, Rt, 0.0)
    asset = (f'<mesh name="asphalt" vertex="{" ".join(map(str, v.ravel()))}" face="{" ".join(map(str, f.ravel()))}"/>')
    wi = R["wet_idx"]
    if len(wi) > 1:
        wv, wf = slab(L[wi], Rt[wi], 0.003)
        wf = wf[[k for k in range(len(wf)) if (wf[k] % len(wi) != len(wi) - 1).all() or True]]
        # 젖은 구간은 열린 띠라 마지막 이음 삼각형을 뺀다
        m = len(wi); keep = []
        for i in range(m - 1):
            j = i + 1
            keep += [[i, m + i, j], [j, m + i, m + j], [2 * m + i, 2 * m + j, 3 * m + i], [2 * m + j, 3 * m + j, 3 * m + i]]
        asset += f'<mesh name="wet" vertex="{" ".join(map(str, wv.ravel()))}" face="{" ".join(map(str, np.array(keep).ravel()))}"/>'
    g = 'contype="0" conaffinity="0"'
    kerbs = ""
    for line in (L, Rt):
        closed = np.r_[line, line[:1]]
        for k, i in enumerate(range(0, len(closed) - 3, 3)):
            p0, p1 = closed[i], closed[i + 3]
            mid, d = (p0 + p1) / 2, p1 - p0
            yaw = math.degrees(math.atan2(d[1], d[0]))
            rgba = "0.88 0.16 0.16 1" if k % 2 else "0.95 0.95 0.95 1"
            kerbs += (f'<geom type="box" pos="{mid[0]} {mid[1]} 0.012" euler="0 0 {yaw}" '
                      f'size="{np.linalg.norm(d) / 2 + 0.01} 0.06 0.012" rgba="{rgba}" {g}/>')
    cx, cy = C[:, 0].mean(), C[:, 1].mean()
    span = float(np.ptp(C, axis=0).max())
    ghosts = "".join(car_xml(f"ghost_{k}", a, b, f"{c[0]} {c[1]} {c[2]} {c[3]}", f"{c[0]} {c[1]} {c[2]} {c[3]}",
                             f"0.1 0.1 0.1 {c[3]}", ghost=True) for k, c in RGBA.items() if k == "truth" or k in models)
    return f"""
<mujoco model="replay">
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
    <map znear="0.01" zfar="500"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.7 0.9" rgb2="0.12 0.16 0.22" width="512" height="512"/>
    <texture name="grass" type="2d" builtin="checker" rgb1="0.16 0.30 0.15" rgb2="0.18 0.33 0.17" width="512" height="512"/>
    <material name="grass" texture="grass" texrepeat="180 180"/>
    {asset}
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 50" dir="0.3 0.3 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom type="plane" size="{3 * span} {3 * span} 0.1" pos="{cx} {cy} -0.005" material="grass" {g}/>
    <geom type="mesh" mesh="asphalt" rgba="0.24 0.25 0.28 1" {g}/>
    {'<geom type="mesh" mesh="wet" rgba="0.18 0.45 0.95 0.55" ' + g + '/>' if len(wi) > 1 else ''}
    {kerbs}
    <camera name="top" pos="{cx} {cy} {span * 0.9}" xyaxes="1 0 0 0 1 0" fovy="60"/>
    {car_xml("car", a, b, "1 0.77 0 1", "0.12 0.12 0.13 1", "0.08 0.08 0.08 1")}
    {ghosts}
  </worldbody>
</mujoco>"""


def heading(path):
    d = path[-1] - path[-4]
    return math.atan2(d[1], d[0])


def quat_z(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def add_line(scene, pts, rgba, width=0.025, z=0.03):
    for p0, p1 in zip(pts[:-1], pts[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        gm = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array([*p0, z]), np.array([*p1, z]))
        scene.ngeom += 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=REPLAY)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--camera", default="chase")
    args = ap.parse_args(argv)
    R = load(args.replay)
    models = [k[5:] for k in R if k.startswith("path_")]
    t0 = time.time()
    m = mujoco.MjModel.from_xml_string(build_model(R, models))
    d = mujoco.MjData(m)
    print(f"모델 빌드 {time.time() - t0:.1f}s, geom {m.ngeom}개")
    rend = mujoco.Renderer(m, 720, 1280, max_geom=20000)
    opt = mujoco.MjvOption()
    jfree = m.joint("car_free").qposadr[0]
    jst = [m.joint(f"car_steer_{w}").qposadr[0] for w in ("fl", "fr")]
    mocap = {k: m.body(f"ghost_{k}").mocapid[0] for k in ["truth", *models]}

    n, frames = len(R["t"]), R["frames"]
    step = int(round(100 / args.fps * 1.0)) or 1
    os.makedirs(OUTDIR, exist_ok=True)
    out_mp4 = os.path.join(OUTDIR, f"mujoco_{args.camera}.mp4")
    w = imageio.get_writer(out_mp4, fps=args.fps, quality=8, macro_block_size=1)
    t1, snap_s = time.time(), int(np.argmin(R["margin"]))
    for s in list(range(frames[0], frames[-1], max(1, round(100 / args.fps)))):
        X, Y, psi, de = (float(R[k][s]) for k in ("X", "Y", "psi", "delta"))
        d.qpos[jfree:jfree + 7] = [X, Y, 0.0, *quat_z(psi)]
        for j in jst:
            d.qpos[j] = de
        fi = int(np.clip(np.searchsorted(frames, s), 0, len(frames) - 1))
        for k in mocap:
            path = R["truth_path"][fi] if k == "truth" else R["path_" + k][fi]
            d.mocap_pos[mocap[k]] = [*path[-1], 0.0]
            d.mocap_quat[mocap[k]] = quat_z(heading(path))
        mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
        rend.update_scene(d, camera=args.camera, scene_option=opt)
        for k in mocap:
            path = R["truth_path"][fi] if k == "truth" else R["path_" + k][fi]
            add_line(rend.scene, path[::4], RGBA[k][:3] + (0.95,), width=0.02 if k != "truth" else 0.03)
        a0 = max(0, s - 300)
        tr = np.c_[R["X"][a0:s + 1:6], R["Y"][a0:s + 1:6]]
        add_line(rend.scene, tr, (1.0, 0.77, 0.0, 0.6), width=0.015, z=0.02)
        img = rend.render()
        w.append_data(img)
        if abs(s - snap_s) < max(1, round(100 / args.fps)):
            imageio.imwrite(os.path.join(OUTDIR, f"mujoco_{args.camera}.png"), img)
    w.close()
    print(f"렌더링 {time.time() - t1:.0f}s -> {out_mp4}")


if __name__ == "__main__":
    main()
