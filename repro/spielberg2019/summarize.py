"""results/*.json을 모아 results.md 표와 figure.png를 만든다."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
NAMES = {"exp_1": "합성: 기본", "exp_2": "합성: 하중 이동", "exp_3": "합성: 타이어 이완", "exp_5": "합성: 두 마찰",
         "exp_6": "합성: 전부", "exp_8": "실데이터: 마른 노면 (원논문 분할)", "dry_runsplit": "실데이터: 마른 노면 (주행 단위 분할)"}

res = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
    r = json.load(open(f))
    res.setdefault(r["exp"], []).append(r)

lines = ["| 실험 | 물리 (우리) | 물리 (원논문) | 신경망 (우리) | 신경망 (원논문) | 물리/신경망 (우리) | 물리/신경망 (원논문) | 피팅된 Cf, Cr, μ |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for e in NAMES:
    if e not in res:
        continue
    rs = res[e]
    b = np.array([r["test_mse"]["bike"] for r in rs]); n = np.array([r["test_mse"]["nn"] for r in rs])
    fmt = lambda v: f"{v.mean():.2e}" + (f" ± {v.std():.1e}" if len(v) > 1 else "")
    o = rs[0].get("original_test_mse")
    pr = rs[0]["bike_params"]
    lines.append(f"| {NAMES[e]} (시드 {len(rs)}) | {fmt(b)} | {o['bike']:.2e} | {fmt(n)} | {o['nn']:.2e} | {b.mean()/n.mean():.1f} | {o['bike']/o['nn']:.1f} | "
                 f"{pr['Cf']/1e3:.0f}k, {pr['Cr']/1e3:.0f}k, {pr['mu']:.2f} |" if o else
                 f"| {NAMES[e]} (시드 {len(rs)}) | {fmt(b)} | - | {fmt(n)} | - | {b.mean()/n.mean():.1f} | - | {pr['Cf']/1e3:.0f}k, {pr['Cr']/1e3:.0f}k, {pr['mu']:.2f} |")
open(os.path.join(ROOT, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))

# 그림: 실험별 test MSE (로그 축), 원논문 대비
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
exps = [e for e in NAMES if e in res]
xs = np.arange(len(exps)); w = 0.2
fig, ax = plt.subplots(figsize=(11, 4.5))
vals = {k: [] for k in ("b", "bo", "n", "no")}
for e in exps:
    rs = res[e]; o = rs[0].get("original_test_mse") or {"bike": np.nan, "nn": np.nan}
    vals["b"].append(np.mean([r["test_mse"]["bike"] for r in rs])); vals["n"].append(np.mean([r["test_mse"]["nn"] for r in rs]))
    vals["bo"].append(o["bike"]); vals["no"].append(o["nn"])
ax.bar(xs - 1.5 * w, vals["bo"], w, label="물리 (원논문)", color="#9db4d6")
ax.bar(xs - 0.5 * w, vals["b"], w, label="물리 (재현)", color="#2f5f9e")
ax.bar(xs + 0.5 * w, vals["no"], w, label="신경망 (원논문)", color="#f0b38a")
ax.bar(xs + 1.5 * w, vals["n"], w, label="신경망 (재현)", color="#c8551b")
ax.set_yscale("log"); ax.set_ylabel("test MSE (다음 시점 r, Uy)")
# 한글 폰트에는 수식용 마이너스 기호가 없어서 눈금을 1e-5 형태로 직접 쓴다
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
ax.set_xticks(xs); ax.set_xticklabels([NAMES[e].replace(": ", "\n").replace(" (", "\n(") for e in exps], fontsize=8)
ax.legend(fontsize=8, ncol=4, loc="upper left"); ax.grid(axis="y", alpha=0.3)
ax.set_title("Spielberg 2019 재현: 물리 모델 vs 신경망")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figure.png"), dpi=150)
print("saved figure.png")
