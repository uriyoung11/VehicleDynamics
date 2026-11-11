"""PNG 연속 사진 폴더를 mp4로. Blender 렌더 결과를 묶을 때 쓴다.

  python -m viz.tools.frames_to_mp4 viz/out/tools/blender viz/out/tools/blender.mp4 --fps 30
"""
import argparse, glob, os
import imageio.v2 as imageio


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("out"); ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.folder, "*.png")))
    w = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=1)
    for f in files:
        w.append_data(imageio.imread(f)[..., :3])
    w.close()
    print(f"{len(files)} 프레임 -> {a.out}")


if __name__ == "__main__":
    main()
