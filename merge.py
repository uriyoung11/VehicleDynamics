import pandas as pd

ROOT = r"C:\Users\이신영\Desktop\KUBIG\d2\code\opti_test"
FILES = {  # 타이어 세트: (파일, 원/테이프)
    "hoons":    ("hoons_all.csv", "plain"),
    "xray":     ("iros_xray_30_7.csv", "plain"),
    "malarska": ("iros_hoons_malarska_23_5.csv", "tape"),
    "duck":     ("xray_duck_test.csv", "tape"),
}
COLS = ["v_x", "v_y", "r", "omega_wheels", "delta", "Iq",
        "ax_imu", "ay_imu", "r_imu", "run_id"]

dfs, offset = [], 0
for tire, (fname, kind) in FILES.items():
    d = pd.read_csv(f"{ROOT}\\{fname}", index_col=0)[COLS].copy()
    d["t"] = d.index
    d["tire"], d["kind"] = tire, kind
    d["run_global"] = d["run_id"] + offset   # 파일 간 run 번호 충돌 방지
    offset = d["run_global"].max() + 1
    dfs.append(d.reset_index(drop=True))

df = pd.concat(dfs, ignore_index=True)
print(df.groupby("tire").agg(rows=("t", "size"), runs=("run_global", "nunique")))