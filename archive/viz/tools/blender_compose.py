"""Blender 렌더(추격 시점 + 위 시점)를 발표용 1920x1080 영상 한 편으로 합친다.

  python -m viz.tools.blender_compose          # viz/out/tools/blender/{chase,top,samples.npy} -> viz/out/tools/blender.mp4

화면 구성
  왼쪽 위   추격 시점 + 상태(속도, 노면 μ, 가장자리 여유) + 범례 + 사건 자막
  오른쪽 위 위에서 본 시점 (차 진행 방향이 항상 위)
  아래      트랙 지도 / 가장자리 여유 그래프 / 모델별 1초 예측 오차 그래프 (현재 시점 세로선)
앞에 제목 화면, 뒤에 요약 화면을 붙인다.
"""
import argparse, os
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W, H = 1920, 1080
BG = (18, 20, 24)
COL = {"truth": (255, 255, 255), "m0": (255, 110, 30), "b1": (40, 190, 255)}
OFF = (217, 0, 255)
NAME = {"truth": "실제 궤적", "m0": "M0 신경망 예측", "b1": "B1 물리 모델 예측"}
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def hexc(c):
    return tuple(v / 255 for v in c)


def fig_to_img(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


# ---------------------------------------------------------------- 아래 띠 (정적 배경 + 좌표 변환)

def strip_static(R, models, t0):
    """트랙 지도와 그래프 두 개를 한 번만 그리고, 매 프레임 커서를 찍을 좌표 변환 함수를 돌려준다."""
    t = R["t"] - t0
    tf = R["t"][R["frames"]] - t0
    fig = plt.figure(figsize=(W / 100, 360 / 100), dpi=100, facecolor=hexc(BG))
    # 트랙 지도
    axm = fig.add_axes([0.005, 0.02, 0.18, 0.88]); axm.set_facecolor(hexc(BG)); axm.set_aspect("equal"); axm.axis("off")
    for k in ("track_left", "track_right"):
        p = np.r_[R[k], R[k][:1]]; axm.plot(p[:, 0], p[:, 1], color="#888", lw=0.8)
    wi = R["wet_idx"]
    axm.plot(R["track_center"][wi, 0], R["track_center"][wi, 1], color="#2a6cff", lw=5, solid_capstyle="butt", label="젖은 구간")
    axm.plot(R["X"], R["Y"], color="#ffc400", lw=1.5)
    axm.set_title("트랙 지도", color="w", fontsize=11, weight="bold", loc="left", pad=2)
    # 가장자리 여유
    wet = R["mu"] < R["mu"].max() - 1e-6
    def style(ax, title):
        ax.set_facecolor("#1c1f25"); ax.tick_params(colors="#bbb", labelsize=9)
        for s in ax.spines.values(): s.set_color("#444")
        ax.set_title(title, color="w", fontsize=11, loc="left", weight="bold", pad=4)
        ax.fill_between(t, 0, 1, where=wet, transform=ax.get_xaxis_transform(), color="#2a6cff", alpha=0.18, lw=0)
        ax.set_xlim(t[R["frames"][0]], t[R["frames"][-1]])
    ax1 = fig.add_axes([0.225, 0.56, 0.765, 0.36]); style(ax1, "트랙 가장자리까지 남은 여유 [m]   (0 아래 = 트랙 이탈, 파란 배경 = 젖은 구간)")
    m = R["margin"]
    ax1.plot(t, m, color="#ddd", lw=1.3)
    ax1.fill_between(t, m, 0, where=m < 0, color=hexc(OFF), alpha=0.8, lw=0)
    ax1.axhline(0, color=hexc(OFF), lw=0.8, ls="--")
    ax1.set_ylim(min(-0.25, m.min() - 0.05), m.max() + 0.1); ax1.tick_params(labelbottom=False)
    ax2 = fig.add_axes([0.225, 0.12, 0.765, 0.32]); style(ax2, "1초 뒤 위치 예측 오차 [m]")
    for k in models:
        e = np.linalg.norm(R["path_" + k][:, -1] - R["truth_path"][:, -1], axis=1)
        ax2.plot(tf, e, color=hexc(COL[k]), lw=1.6, label=NAME[k].replace(" 예측", ""))
    ax2.set_ylim(0, None); ax2.set_xlabel("시간 [s]", color="#bbb", fontsize=9, labelpad=1)
    ax2.legend(loc="upper left", fontsize=9, facecolor="#1c1f25", edgecolor="#444", labelcolor="w")
    img = fig_to_img(fig)
    hgt = img.shape[0]

    def to_px(ax, x, y):
        X, Y = ax.transData.transform((x, y)); return X, hgt - Y
    ys = [to_px(a, 0, a.get_ylim()[k])[1] for a in (ax1, ax2) for k in (0, 1)]
    plt.close(fig)
    return img, lambda tt: to_px(ax1, tt, 0)[0], (min(ys), max(ys)), lambda x, y: to_px(axm, x, y)


# ---------------------------------------------------------------- 글자 상자

def box(dr, xy, wh, alpha=170):
    x, y = xy; dr.rounded_rectangle([x, y, x + wh[0], y + wh[1]], 12, fill=(10, 12, 16, alpha))


def banner(dr, text, color, y=24, cx=640):
    f = font(34, True); w = dr.textlength(text, font=f)
    box(dr, (cx - w / 2 - 22, y), (w + 44, 62), 200)
    dr.rounded_rectangle([cx - w / 2 - 22, y, cx + w / 2 + 22, y + 62], 12, outline=color, width=3)
    dr.text((cx - w / 2, y + 7), text, font=f, fill=color)


def legend(dr, models, x, y):
    items = ["truth", *models]
    box(dr, (x, y), (290, 22 + 34 * (len(items) + 1)))
    f = font(20)
    for i, k in enumerate(items):
        yy = y + 16 + 34 * i
        dr.line([x + 16, yy + 13, x + 66, yy + 13], fill=COL[k], width=7 if k == "truth" else 5)
        dr.text((x + 80, yy), NAME[k], font=f, fill=(235, 235, 235))
    yy = y + 16 + 34 * len(items)
    for j, c in enumerate([(26, 217, 64), (255, 191, 0), OFF]):
        dr.line([x + 16 + 17 * j, yy + 13, x + 31 + 17 * j, yy + 13], fill=c, width=7)
    dr.text((x + 80, yy), "지나온 길 (여유→이탈)", font=f, fill=(235, 235, 235))


# ---------------------------------------------------------------- 제목·요약 화면

def title_card(models):
    im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    dr.text((W / 2, 330), "신경망 차량 모델로 한계 주행하기", font=font(72, True), fill="white", anchor="mm")
    dr.text((W / 2, 430), "젖은 노면에서 차가 미끄러지는 순간, 모델은 1초 뒤 위치를 얼마나 맞힐까?",
            font=font(36), fill=(200, 205, 215), anchor="mm")
    lines = [("노란 차", "실제 주행 (물리 시뮬레이션, 한계 근처 속도)", (255, 196, 0)),
             ("파란 노면", "젖은 구간: 마찰계수 μ 0.70 → 0.55", (42, 108, 255)),
             *[(NAME[k].split()[0], f"{NAME[k]} — 매 순간 1초 앞을 예측한 궤적", COL[k]) for k in models]]
    for i, (a, b, c) in enumerate(lines):
        y = 560 + 62 * i
        dr.rounded_rectangle([560, y, 590, y + 30], 6, fill=c)
        dr.text((612, y - 4), b, font=font(30), fill=(225, 225, 225))
    return np.asarray(im)


def summary_card(R, models):
    im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    dr.text((W / 2, 200), "예측 오차 요약", font=font(64, True), fill="white", anchor="mm")
    T = R["truth_path"]; fs = R["frames"]
    cols = [("모델", 640), ("0.5초 뒤 오차", 960), ("1초 뒤 오차", 1220), ("1초 뒤 최대 오차", 1520)]
    for name, x in cols:
        dr.text((x, 340), name, font=font(32, True), fill=(180, 185, 195), anchor="mm")
    dr.line([380, 380, 1700, 380], fill=(80, 80, 90), width=2)
    for i, k in enumerate(models):
        P = R["path_" + k]
        e5 = np.linalg.norm(P[:, 50] - T[:, 50], axis=1); e10 = np.linalg.norm(P[:, -1] - T[:, -1], axis=1)
        y = 450 + 90 * i
        dr.text((640, y), NAME[k].replace(" 예측", ""), font=font(36, True), fill=COL[k], anchor="mm")
        for v, x in ((e5.mean(), 960), (e10.mean(), 1220), (e10.max(), 1520)):
            dr.text((x, y), f"{v * 100:.1f} cm", font=font(40), fill="white", anchor="mm")
    dr.text((W / 2, 760), "오차 = 예측 궤적 끝점과 실제 끝점 사이 거리. 이 장면 하나 기준 (평균 / 최대)",
            font=font(26), fill=(150, 155, 165), anchor="mm")
    car = f"차 크기: 축간거리 {float(R['car_a'] + R['car_b']) * 100:.0f} cm (1/10 스케일 F1TENTH)"
    dr.text((W / 2, 810), car, font=font(26), fill=(150, 155, 165), anchor="mm")
    return np.asarray(im)


# ---------------------------------------------------------------- 본편

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "viz", "out", "tools", "blender"))
    ap.add_argument("--replay", default=os.path.join(ROOT, "viz", "out", "replay.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "viz", "out", "tools", "blender.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--slowmo", type=float, default=0.35)
    ap.add_argument("--still", type=int, default=None, help="이 프레임 하나만 PNG로 (시험용)")
    ap.add_argument("--cards", action="store_true", help="제목·요약 화면만 PNG로 (시험용)")
    a = ap.parse_args(argv)
    R = dict(np.load(a.replay))
    models = [k[5:] for k in R if k.startswith("path_")]
    if a.cards:
        Image.fromarray(title_card(models)).save(a.out.replace(".mp4", "_title.png"))
        Image.fromarray(summary_card(R, models)).save(a.out.replace(".mp4", "_summary.png")); return
    samples = np.load(os.path.join(a.src, "samples.npy"))
    t0 = R["t"][samples[0]]
    strip, t2x, (gy0, gy1), map_px = strip_static(R, models, t0)
    mu_hi = R["mu"].max()
    wet_on = int(np.argmax(R["mu"] < mu_hi - 1e-6))
    nf = len(os.listdir(os.path.join(a.src, "chase")))
    idx = range(nf) if a.still is None else [a.still]

    if a.still is None:
        w = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=1)
        for img, sec in ((title_card(models), 4.0),):
            for _ in range(int(sec * a.fps)): w.append_data(img)
    for fno in idx:
        s = int(samples[fno])
        fi = int(np.clip(np.searchsorted(R["frames"], s), 0, len(R["frames"]) - 1))
        im = Image.new("RGB", (W, H), BG)
        im.paste(Image.open(os.path.join(a.src, "chase", f"f{fno:05d}.png")).convert("RGB"), (0, 0))
        im.paste(Image.open(os.path.join(a.src, "top", f"f{fno:05d}.png")).convert("RGB"), (1280, 0))
        im.paste(Image.fromarray(strip), (0, 720))
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
        # 아래 그래프 커서, 지도 위 차 위치
        x = t2x(R["t"][s] - t0)
        dr.line([x, 720 + gy0, x, 720 + gy1], fill=(255, 255, 255, 220), width=2)
        mx, my = map_px(R["X"][s], R["Y"][s])
        dr.ellipse([mx - 8, 720 + my - 8, mx + 8, 720 + my + 8], fill=(255, 196, 0, 255), outline="white", width=2)
        # 상태 상자
        m, mu = float(R["margin"][s]), float(R["mu"][s])
        box(dr, (20, 20), (330, 176))
        f, fb = font(24), font(24, True)
        dr.text((38, 32), f"시간  {R['t'][s] - t0:5.2f} s", font=f, fill="white")
        dr.text((38, 68), f"속도  {R['ux'][s] * 3.6:5.1f} km/h", font=f, fill="white")
        dr.text((38, 104), f"노면  μ {mu:.2f} " + ("(젖음)" if mu < mu_hi - 1e-6 else "(마름)"), font=f,
                fill=(110, 160, 255) if mu < mu_hi - 1e-6 else "white")
        mc = OFF if m < 0 else ((255, 191, 0) if m < 0.3 else (26, 217, 64))
        dr.text((38, 140), f"여유  {m * 100:+5.0f} cm" + ("  이탈" if m < 0 else ""), font=fb, fill=mc)
        legend(dr, models, 970, 20)
        # 위 시점 이름표와 1 m 눈금 (ortho_scale 6.5 m가 세로 720 px)
        box(dr, (1296, 16), (300, 44))
        dr.text((1310, 22), "위에서 본 시점 (진행 방향 ↑)", font=font(20), fill="white")
        px_m = 720 / 6.5
        dr.line([1300, 700, 1300 + px_m, 700], fill="white", width=4)
        dr.text((1300 + px_m + 10, 686), "1 m", font=font(20), fill="white")
        # 모델별 현재 1초 오차 (위 시점 오른쪽 아래)
        box(dr, (1660, 590), (244, 22 + 34 * len(models)))
        for i, k in enumerate(models):
            e = float(np.linalg.norm(R["path_" + k][fi, -1] - R["truth_path"][fi, -1]))
            dr.text((1676, 600 + 34 * i), f"{k.upper()} 1초 오차 {e * 100:4.0f} cm", font=font(20, True), fill=COL[k])
        # 사건 자막 (이탈 > 예측 크게 빗나감 > 젖은 노면 진입)
        errs = {k: float(np.linalg.norm(R["path_" + k][fi, -1] - R["truth_path"][fi, -1])) for k in models}
        worst = max(errs, key=errs.get)
        if m < 0:
            banner(dr, f"트랙 이탈!  가장자리 밖 {-m * 100:.0f} cm", OFF)
        elif errs[worst] > 0.3:
            banner(dr, f"{NAME[worst].split()[0]} 예측이 크게 빗나감  1초 뒤 {errs[worst] * 100:.0f} cm", COL[worst])
        elif 0 <= s - wet_on < 150:
            banner(dr, f"젖은 노면 진입  μ {mu_hi:.2f} → {mu:.2f}", (110, 160, 255))
        if fno + 1 < len(samples) and samples[fno + 1] - s < 100 / a.fps * 0.8:
            dr.text((640, 690), f"슬로모션 ×{a.slowmo:.2f}", font=font(22, True), fill=(255, 255, 255, 200), anchor="mm")
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        if a.still is not None:
            p = a.out.replace(".mp4", f"_still{fno}.png"); im.save(p); print("saved", p); return
        w.append_data(np.asarray(im))
    last = np.asarray(im)
    for _ in range(int(1.0 * a.fps)): w.append_data(last)
    for _ in range(int(5.0 * a.fps)): w.append_data(summary_card(R, models))
    w.close()
    print(f"{nf} 프레임 + 제목/요약 -> {a.out}")


if __name__ == "__main__":
    main()
