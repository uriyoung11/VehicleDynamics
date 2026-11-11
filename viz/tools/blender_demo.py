"""시각화 도구 시험 3: Blender. 공용 장면(viz/out/replay.npz)을 Blender로 렌더링해 PNG 연속 사진으로 저장한다.

  blender -b -P viz/tools/blender_demo.py -- --fps 30          # Blender 4.2 이상 -> out/chase, out/top, out/samples.npy
  python -m viz.tools.blender_compose                            # 두 시점 + 지도 + 그래프 -> blender.mp4

시점 두 개를 같은 프레임마다 렌더링한다.
  chase : 차 뒤에서 따라가는 시점
  top   : 위에서 내려다본 시점. 차 진행 방향이 항상 화면 위쪽이라 앞으로 1초 궤적이 잘 보인다.
트랙 이탈 순간 앞뒤는 느리게 재생한다(--slowmo).

Blender 안의 파이썬에는 torch 등이 없으므로 replay.npz만 읽는다. Blender 물리는 쓰지 않는다.
--still N : N번째 프레임 하나만 렌더링(시험용). --blend : 장면을 .blend로 저장해 GUI에서 열어 볼 수 있게.
"""
import argparse, math, os, sys, time
import numpy as np
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COL = {"truth": (1, 1, 1), "m0": (1.0, 0.3, 0.0), "b1": (0.0, 0.75, 1.0)}
OFF = (0.85, 0.0, 1.0)   # 트랙 이탈 색(보라). 빨간 연석과 겹치지 않게


def args_():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=os.path.join(ROOT, "viz", "out", "replay.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "viz", "out", "tools", "blender"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--engine", default="eevee", choices=["eevee", "cycles", "workbench"])
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--top-res", type=int, nargs=2, default=[640, 720])
    ap.add_argument("--views", nargs="+", default=["chase", "top"])
    ap.add_argument("--slowmo", type=float, default=0.35, help="이탈 순간 재생 속도 (1 = 실시간)")
    ap.add_argument("--still", type=int, default=None, help="이 프레임 하나만 (0 = 트랙 이탈 순간)")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="앞에서부터 이만큼만 (시험용)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- 재질

def mat(name, rgb, rough=0.6, metal=0.0, alpha=1.0, emit=0.0, clearcoat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Alpha"].default_value = alpha
    if clearcoat:
        b.inputs["Coat Weight"].default_value = clearcoat
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1:
        m.surface_render_method = "BLENDED"
    return m


def mesh_obj(name, verts, faces, material):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(material)
    return ob


def strip(left, right, z, closed=True):
    n = len(left)
    v = np.vstack([np.c_[left, np.full(n, z)], np.c_[right, np.full(n, z)]])
    f = [(i, i + 1 if i + 1 < n else 0, n + (i + 1 if i + 1 < n else 0), n + i) for i in range(n if closed else n - 1)]
    return v, f


def offset_line(line, other, d):
    v = line - other
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return line + d * v


# ---------------------------------------------------------------- 장면

def build_track(R):
    L, Rt = R["track_left"], R["track_right"]
    v, f = strip(L, Rt, 0.0)
    mesh_obj("asphalt", v, f, mat("asphalt", (0.035, 0.036, 0.04), rough=0.8))
    wi = R["wet_idx"]
    if len(wi) > 1:
        v, f = strip(L[wi], Rt[wi], 0.006, closed=False)
        mesh_obj("wet", v, f, mat("wet", (0.01, 0.04, 0.16), rough=0.03, alpha=0.9, clearcoat=1.0))
    # 연석: 경계 바깥 12 cm 띠, 3칸마다 빨강/흰색
    red, white = mat("kerb_red", (0.6, 0.03, 0.03), rough=0.5), mat("kerb_white", (0.85, 0.85, 0.85), rough=0.5)
    for side, line, other in (("l", L, Rt), ("r", Rt, L)):
        outer = offset_line(line, other, 0.12)
        v, f = strip(line, outer, 0.012)
        ob = mesh_obj(f"kerb_{side}", v, f, red)
        ob.data.materials.append(white)
        for i, p in enumerate(ob.data.polygons):
            p.material_index = (i // 3) % 2
    # 잔디
    C = R["track_center"]
    cx, cy = C.mean(0); span = float(np.ptp(C, axis=0).max()) * 2
    bpy.ops.mesh.primitive_plane_add(size=span, location=(cx, cy, -0.03))
    g = bpy.context.object; g.name = "grass"
    gm = mat("grass", (0.03, 0.09, 0.025), rough=0.95)
    tex = gm.node_tree.nodes.new("ShaderNodeTexNoise"); tex.inputs["Scale"].default_value = 12; tex.inputs["Detail"].default_value = 1.0  # 잘면 먼 곳이 프레임마다 반짝인다
    ramp = gm.node_tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.025, 0.07, 0.02, 1); ramp.color_ramp.elements[1].color = (0.05, 0.13, 0.04, 1)
    gm.node_tree.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    gm.node_tree.links.new(ramp.outputs["Color"], gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    g.data.materials.append(gm)


def add_box(name, size, loc, material, parent, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    ob = bpy.context.object; ob.name = name; ob.scale = size
    if bevel:
        mod = ob.modifiers.new("bevel", "BEVEL"); mod.width = bevel; mod.segments = 4
        bpy.ops.object.transform_apply(scale=True)
    ob.data.materials.append(material); ob.parent = parent
    bpy.ops.object.shade_smooth()
    return ob


def add_wheel(name, loc, parent, tire, rim):
    e = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(e)
    e.location = loc; e.parent = parent
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.045, vertices=32, rotation=(math.pi / 2, 0, 0))
    t = bpy.context.object; t.name = name + "_tire"; t.parent = e; t.data.materials.append(tire)
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.047, vertices=24, rotation=(math.pi / 2, 0, 0))
    r = bpy.context.object; r.name = name + "_rim"; r.parent = e; r.data.materials.append(rim)
    return e


def blob_shadow(name, size, loc, parent, strength=0.95):
    """차 밑 그림자: 가운데가 진하고 가장자리로 갈수록 투명해지는 판."""
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
    ob = bpy.context.object; ob.name = name; ob.scale = (*size, 1); ob.parent = parent
    m = bpy.data.materials.new(name + "_m"); m.use_nodes = True; m.surface_render_method = "DITHERED"
    nt = m.node_tree; bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1); bsdf.inputs["Roughness"].default_value = 1.0
    # 판 UV(0~1)에서 중심까지 거리 → 가운데 진하고 가장자리 투명
    uv = nt.nodes.new("ShaderNodeUVMap")
    sub = nt.nodes.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"; sub.inputs[1].default_value = (0.5, 0.5, 0)
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.2; ramp.color_ramp.elements[0].color = (strength, strength, strength, 1)
    ramp.color_ramp.elements[1].position = 0.5; ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
    L = nt.links.new
    L(uv.outputs["UV"], sub.inputs[0]); L(sub.outputs[0], ln.inputs[0]); L(ln.outputs["Value"], ramp.inputs["Fac"])
    L(ramp.outputs["Color"], bsdf.inputs["Alpha"])
    ob.data.materials.append(m)
    return ob


def build_car(R, name, body_mat, ghost=False):
    a, b = float(R["car_a"]), float(R["car_b"])
    root = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(root)
    cx = (a - b) / 2
    if not ghost:
        blob_shadow(name + "_shadow", (a + b + 0.36, 0.42), (cx + 0.04, -0.03, 0.016), root)
    add_box(name + "_body", (a + b + 0.18, 0.23, 0.05), (cx, 0, 0.065), body_mat, root, bevel=0.02)
    if ghost:
        return root, None
    shell = mat(name + "_shell", (0.02, 0.02, 0.025), rough=0.15, clearcoat=1.0)
    add_box(name + "_shell", (0.26, 0.18, 0.05), (cx - 0.02, 0, 0.11), shell, root, bevel=0.02)
    add_box(name + "_wing", (0.05, 0.25, 0.012), (-b - 0.08, 0, 0.15), shell, root, bevel=0.004)
    add_box(name + "_nose", (0.04, 0.24, 0.02), (a + 0.11, 0, 0.045), mat(name + "_red", (0.7, 0.03, 0.03), rough=0.3), root, bevel=0.006)
    tire, rim = mat("tire", (0.015, 0.015, 0.015), rough=0.9), mat("rim", (0.7, 0.7, 0.72), rough=0.25, metal=1.0)
    steer = [add_wheel(f"{name}_w{w}", (x, y, 0.05), root, tire, rim)
             for w, x, y in (("fl", a, 0.14), ("fr", a, -0.14), ("rl", -b, 0.14), ("rr", -b, -0.14))][:2]
    return root, steer


def make_curve(name, npts, rgb, radius, emit):
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = radius; cu.bevel_resolution = 3
    sp = cu.splines.new("POLY"); sp.points.add(npts - 1)
    ob = bpy.data.objects.new(name, cu); bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat(name + "_m", rgb, rough=0.3, emit=emit))
    return ob


def set_curve(ob, xy, z):
    pts = ob.data.splines[0].points
    idx = np.linspace(0, len(xy) - 1, len(pts)).round().astype(int)
    for p, i in zip(pts, idx):
        p.co = (float(xy[i, 0]), float(xy[i, 1]), z, 1.0)


def margin_rgb(m):
    """트랙 가장자리까지 여유 -> 색. 0.3 m 이상 초록, 0에 가까울수록 노랑, 이탈(음수)은 보라."""
    m = np.asarray(m)
    k = np.clip(m / 0.3, 0, 1)[:, None]
    c = (1 - k) * np.array([1.0, 0.75, 0.0]) + k * np.array([0.1, 0.85, 0.25])
    c[m < 0] = OFF
    return c


def ribbon_mat():
    m = bpy.data.materials.new("trail_m"); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "col"
    nt.links.new(at.outputs["Color"], b.inputs["Base Color"]); nt.links.new(at.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 0.6; b.inputs["Roughness"].default_value = 0.5
    return m


def set_ribbon(ob, xy, rgb, width, z):
    """지나온 길을 바닥에 붙은 띠로. 점마다 색이 다르다."""
    old = ob.data
    t = np.gradient(xy, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.c_[-t[:, 1], t[:, 0]] * width / 2
    n = len(xy)
    v = np.vstack([np.c_[xy + nrm, np.full(n, z)], np.c_[xy - nrm, np.full(n, z)]])
    f = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
    me = bpy.data.meshes.new("trail")
    me.from_pydata([tuple(p) for p in v], [], f); me.update()
    at = me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
    cols = np.vstack([rgb, rgb])
    at.data.foreach_set("color", np.c_[cols, np.ones(2 * n)].ravel().astype(np.float32))
    me.materials.append(old.materials[0])
    ob.data = me
    bpy.data.meshes.remove(old)


def top_camera(args):
    """위에서 내려다보는 정사영 카메라. 매 프레임 차를 따라가고 차 진행 방향을 화면 위로 돌린다."""
    cd = bpy.data.cameras.new("top"); cd.type = "ORTHO"; cd.ortho_scale = 6.5
    cd.clip_start = 0.1; cd.clip_end = 50
    cam = bpy.data.objects.new("top", cd); bpy.context.collection.objects.link(cam)
    return cam


def smooth_heading(psi, win=80):
    u = np.unwrap(psi); k = np.ones(win) / win
    return np.convolve(np.pad(u, win // 2, mode="edge"), k, mode="same")[win // 2:win // 2 + len(u)]


def sample_schedule(R, fps, slow):
    """재생할 시뮬레이션 샘플 번호. 이탈 순간(margin 최소) 앞뒤 ±1.2초는 slow배 속도로."""
    frames = R["frames"]; ev = int(np.argmin(R["margin"]))
    s, out = float(frames[0]), []
    while s < frames[-1]:
        out.append(int(round(s)))
        d = abs(s - ev) / 100
        speed = slow if d < 1.2 else (1.0 if d > 1.8 else slow + (1 - slow) * (d - 1.2) / 0.6)
        s += 100 / fps * speed
    return out


def setup_world_camera(car, args):
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = args.res
    eng = {"eevee": "BLENDER_EEVEE_NEXT", "cycles": "CYCLES", "workbench": "BLENDER_WORKBENCH"}[args.engine]
    sc.render.engine = eng
    if args.engine == "cycles":
        sc.cycles.samples = 32; sc.cycles.use_denoising = True; sc.cycles.device = "CPU"
    if args.engine == "eevee":
        sc.eevee.taa_render_samples = 32
        sc.eevee.use_shadows = True

    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = -0.9
    w = bpy.data.worlds.new("world"); sc.world = w; w.use_nodes = True
    nt = w.node_tree; sky = nt.nodes.new("ShaderNodeTexSky"); sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(35); sky.sun_rotation = math.radians(60)
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 0.35
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(50), 0, math.radians(150)))
    sun = bpy.context.object; sun.data.energy = 3.5; sun.data.angle = math.radians(2)
    # 햇빛 그림자는 끈다. 넓은 바닥에서 EEVEE 그림자 타일이 빠졌다 들어왔다 하며 네모나게 번쩍였다.
    # 차 그림자는 차 밑에 붙인 흐릿한 판(blob_shadow)으로 대신한다.
    sun.data.use_shadow = False
    cam_data = bpy.data.cameras.new("chase"); cam_data.lens = 24; cam_data.clip_start = 0.05; cam_data.clip_end = 250
    cam = bpy.data.objects.new("chase", cam_data); bpy.context.collection.objects.link(cam)
    cam.parent = car; cam.location = (-1.9, 0.0, 0.85)
    cam.rotation_euler = (math.radians(68), 0, math.radians(-90))
    sc.camera = cam


def main():
    args = args_()
    args.out = os.path.abspath(args.out)
    R = dict(np.load(args.replay))
    models = [k[5:] for k in R if k.startswith("path_")]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build_track(R)
    car, steer = build_car(R, "car", mat("car_body", (1.0, 0.55, 0.0), rough=0.25, clearcoat=1.0))
    ghosts = {k: build_car(R, f"ghost_{k}", mat(f"ghost_{k}_m", COL[k], rough=0.3, alpha=0.5 if k != "truth" else 0.3,
                                                 emit=0.4), ghost=True)[0] for k in ["truth", *models]}
    # 실제(흰색)는 굵게 아래에, 모델 예측은 가늘게 위에 그려 겹쳐도 보이게
    paths = {k: make_curve(f"path_{k}", 41, COL[k], 0.04 if k == "truth" else 0.022, 0.8 if k == "truth" else 1.5)
             for k in ["truth", *models]}
    trail = mesh_obj("trail", [(0, 0, 0)] * 3, [(0, 1, 2)], ribbon_mat())
    setup_world_camera(car, args)
    chase, top = bpy.context.scene.camera, top_camera(args)
    psi_s = smooth_heading(R["psi"])
    margin_c = margin_rgb(R["margin"])

    frames = R["frames"]
    samples = sample_schedule(R, args.fps, args.slowmo)
    if args.still is not None:
        samples = [int(np.argmin(R["margin"])) if args.still == 0 else samples[args.still]]
    if args.max_frames:
        k = int(np.searchsorted(samples, int(np.argmin(R["margin"]))))
        samples = samples[max(0, k - args.max_frames // 2):][:args.max_frames]
    for v in args.views:
        os.makedirs(os.path.join(args.out, v), exist_ok=True)
    np.save(os.path.join(args.out, "samples.npy"), np.array(samples))
    sc = bpy.context.scene
    t_all = time.time()
    # 시점마다 영상 전체를 따로 한 번씩 그린다. 한 프레임 안에서 카메라·해상도를 번갈아 바꾸면
    # Blender가 매번 렌더 준비를 처음부터 다시 해서(약 70초) 10배 넘게 느려진다.
    # 이미 있는 PNG는 건너뛰므로 중간에 멈춰도 이어서 그릴 수 있다.
    for v in args.views:
        sc.camera = chase if v == "chase" else top
        sc.render.resolution_x, sc.render.resolution_y = args.res if v == "chase" else args.top_res
        t0, done = time.time(), 0
        for fno, s in enumerate(samples):
            fp = os.path.join(args.out, v, f"f{fno:05d}.png")
            if os.path.exists(fp) and args.still is None:
                continue
            X, Y, psi, de = (float(R[k][s]) for k in ("X", "Y", "psi", "delta"))
            car.location = (X, Y, 0); car.rotation_euler = (0, 0, psi)
            for w in steer:
                w.rotation_euler = (0, 0, de)
            fi = int(np.clip(np.searchsorted(frames, s), 0, len(frames) - 1))
            for k in ghosts:
                p = R["truth_path"][fi] if k == "truth" else R["path_" + k][fi]
                d = p[-1] - p[-4]
                ghosts[k].location = (p[-1, 0], p[-1, 1], 0); ghosts[k].rotation_euler = (0, 0, math.atan2(d[1], d[0]))
                set_curve(paths[k], p, 0.02 if k == "truth" else 0.05 + 0.01 * list(ghosts).index(k))
            idx = np.arange(max(0, s - 400), s + 1, 2)
            if len(idx) > 2:
                set_ribbon(trail, np.c_[R["X"][idx], R["Y"][idx]], margin_c[idx], 0.09, 0.009)
            # 위 카메라: 차가 화면 아래쪽 1/4 지점에 오도록 진행 방향으로 2 m 앞을 비춘다
            h = float(psi_s[s])
            top.location = (X + 2.0 * math.cos(h), Y + 2.0 * math.sin(h), 20.0)
            top.rotation_euler = (0, 0, h - math.pi / 2)
            sc.render.filepath = fp
            bpy.ops.render.render(write_still=True)
            done += 1
            if done <= 3 or done % 30 == 0:
                print(f"[blender] {v} {fno + 1}/{len(samples)} 프레임, {(time.time() - t0) / done:.2f}s/프레임", flush=True)
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.dirname(args.out), "blender_scene.blend"))
    print(f"[blender] 완료 {time.time() - t_all:.0f}s -> {args.out}")


main()
