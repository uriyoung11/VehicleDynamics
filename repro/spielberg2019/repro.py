"""Spielberg et al. 2019 (Science Robotics) 모델 학습 실험의 PyTorch 재현.

원본: data/scirob_submission/Model_Learning/{models.py, train_models.py} (TF1)
원본과 맞춘 것:
  - NN: 20개 입력(4시점 x [r, Uy, Ux, delta, Fxf/FX_NORM]) -> 128 softplus -> 128 softplus -> 2
        예측 = 마지막 시점 [r, Uy] + DT * NN 출력
  - Bike: Fiala 타이어 자전거 모델, 마지막 시점만 사용. Cf, Cr, mu 배율만 학습(초기값 N(1.1, 0.1)), m, Izz 고정
  - 손실: 다음 시점 [r, Uy] 상태의 MSE (두 출력 평균)
  - Adam lr 1e-4 (eps 1e-7, TF 기본값), 배치 1000, 에폭 1500, 매 에폭 같은 순서(셔플 없음)
  - 가중치 초기화: glorot uniform, bias 0 (TF dense 기본값)

사용 예:
  python repro.py --exp exp_8                 # 원본 dry.npz 분할(샘플 단위 셔플) 그대로
  python repro.py --exp dry_runsplit          # 원본 주행 파일에서 주행 단위로 분할
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(ROOT, "..", "..", "data", "scirob_submission", "Model_Learning")

# learning_params.py 값
VEH = dict(a=1.194, b=1.437, m=1776.2, Izz=2763.49, Cf=150.0e3, Cr=170.0e3, mu=1.0, g=9.81)
DT, T, NS = 0.01, 4, 5
FX_NORM = VEH["m"] * 100

ORIG_FILES = {  # 원본 결과와 비교할 실험
    "exp_1": "data/gen/exp_1.npz", "exp_2": "data/gen/exp_2.npz", "exp_3": "data/gen/exp_3.npz",
    "exp_5": "data/gen/exp_5.npz", "exp_6": "data/gen/exp_6.npz", "exp_8": "data/exp/exp_data/dry.npz",
}
ORIG_RESULTS = {"exp_1": "gen/exp_1", "exp_2": "gen/exp_2", "exp_3": "gen/exp_3",
                "exp_5": "gen/exp_5", "exp_6": "gen/exp_6", "exp_8": "exp/exp_8"}

# 주행 단위 분할: 고속 타원 주행(exp_2,8,9,10,16)과 저속 큰 조향 주행(exp_4,5,6,11,12,15)이 섞이도록 고름
RUN_SPLIT = {"dev": ["exp_11", "exp_12", "exp_15"], "test": ["exp_5", "exp_9"]}
TRANSIENT = 20  # 각 주행 앞부분의 필터 시작 구간(ux가 0.9 -> 30 m/s로 튐) 제거


def windows(z):
    F = np.stack([z["r"], z["uy"], z["ux"], z["delta"], z["fxf"] / FX_NORM], 1)[TRANSIENT:]
    n = len(F) - T
    X = np.stack([F[i:i + T].ravel() for i in range(n)])
    Y = np.stack([F[T:, 0], F[T:, 1]], 1)
    return X, Y


def load(exp):
    if exp in ORIG_FILES:
        d = np.load(os.path.join(ML, ORIG_FILES[exp]))
        return {s: (d[s + "_f"], d[s + "_t"]) for s in ["train", "dev", "test"]}
    if exp == "dry_runsplit":
        out = {"train": [], "dev": [], "test": []}
        for f in sorted(glob.glob(os.path.join(ML, "data/exp/exp_data/dry/*.npz"))):
            name = os.path.basename(f)[:-4]
            split = next((s for s, runs in RUN_SPLIT.items() if name in runs), "train")
            out[split].append(windows(np.load(f)))
        return {s: (np.concatenate([x for x, _ in v]), np.concatenate([y for _, y in v])) for s, v in out.items()}
    raise ValueError(exp)


class NNModel(nn.Module):
    def __init__(self, n1=128, n2=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NS * T, n1), nn.Softplus(), nn.Linear(n1, n2), nn.Softplus(), nn.Linear(n2, 2))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        last = x[:, NS * (T - 1):NS * (T - 1) + 2]
        return last + DT * self.net(x)


class BikeModel(nn.Module):
    def __init__(self, rng):
        super().__init__()
        init = lambda: nn.Parameter(torch.tensor(rng.normal(1.1, 0.1), dtype=torch.float64))
        # 원본은 m, I도 변수로 만들지만 실제 계산에는 상수를 씀
        self.Ca_f_norm, self.Ca_r_norm, self.mu_norm = init(), init(), init()

    def params(self):
        return dict(Cf=(self.Ca_f_norm * VEH["Cf"]).item(), Cr=(self.Ca_r_norm * VEH["Cr"]).item(), mu=(self.mu_norm * VEH["mu"]).item())

    @staticmethod
    def fiala(alpha, Ca, mu, fz):
        alpha_slide = torch.abs(torch.atan(3 * mu * fz / Ca))
        ta = torch.tan(alpha)
        lin = -Ca * ta + (Ca ** 2) / (3 * mu * fz) * torch.abs(ta) * ta - (Ca ** 3) / (9 * mu ** 2 * fz ** 2) * ta ** 3 * (1 - 2 * mu / (3 * mu))
        return torch.where(torch.abs(alpha) < alpha_slide, lin, -mu * fz * torch.sign(alpha))

    def forward(self, x):
        a, b, m, I, g = VEH["a"], VEH["b"], VEH["m"], VEH["Izz"], VEH["g"]
        o = NS * (T - 1)
        r, uy, ux, d, fxf = x[:, o], x[:, o + 1], x[:, o + 2], x[:, o + 3], x[:, o + 4] * FX_NORM
        Cf, Cr, mu = self.Ca_f_norm * VEH["Cf"], self.Ca_r_norm * VEH["Cr"], self.mu_norm * VEH["mu"]
        L = a + b
        fzf, fzr = m * b * g / L, m * a * g / L
        af = torch.atan((uy + a * r) / ux) - d
        ar = torch.atan((uy - b * r) / ux)
        fyf, fyr = self.fiala(af, Cf, mu, fzf), self.fiala(ar, Cr, mu, fzr)
        r_dot = (a * fyf * torch.cos(d) + a * fxf * torch.sin(d) - b * fyr) / I
        uy_dot = (fyf * torch.cos(d) + fxf * torch.sin(d) + fyr) / m - r * ux
        return torch.stack([r + DT * r_dot, uy + DT * uy_dot], 1)


def mse(model, X, Y):
    with torch.no_grad():
        return float(((model(X) - Y) ** 2).mean())


def mse_per_target(model, X, Y):
    with torch.no_grad():
        return ((model(X) - Y) ** 2).mean(0).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    data = {s: tuple(torch.tensor(a, dtype=torch.float64) for a in v) for s, v in load(args.exp).items()}
    Xtr, Ytr = data["train"]
    nn_m, bike = NNModel().double(), BikeModel(rng)
    opt_nn = torch.optim.Adam(nn_m.parameters(), lr=1e-4, eps=1e-7)
    opt_b = torch.optim.Adam(bike.parameters(), lr=1e-4, eps=1e-7)
    B = 1000
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        for i in range(0, len(Xtr), B):  # 원본: len//B + 1 배치 = 한 번 순회, 셔플 없음
            xb, yb = Xtr[i:i + B], Ytr[i:i + B]
            for model, opt in ((nn_m, opt_nn), (bike, opt_b)):
                opt.zero_grad(); loss = ((model(xb) - yb) ** 2).mean(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            row = dict(epoch=ep, train=[mse(bike, Xtr, Ytr), mse(nn_m, Xtr, Ytr)], dev=[mse(bike, *data["dev"]), mse(nn_m, *data["dev"])])
            hist.append(row)
            print(f"[{args.exp} s{args.seed}] ep {ep:4d} {time.time()-t0:6.0f}s bike train {row['train'][0]:.3e} dev {row['dev'][0]:.3e} | nn train {row['train'][1]:.3e} dev {row['dev'][1]:.3e}", flush=True)

    res = dict(exp=args.exp, seed=args.seed, epochs=args.epochs, n={s: len(v[0]) for s, v in data.items()},
               test_mse=dict(bike=mse(bike, *data["test"]), nn=mse(nn_m, *data["test"])),
               test_mse_per_target=dict(bike=mse_per_target(bike, *data["test"]), nn=mse_per_target(nn_m, *data["test"])),
               bike_params=bike.params(), history=hist, seconds=time.time() - t0)
    if args.exp in ORIG_RESULTS:
        o = np.load(os.path.join(ML, "results", ORIG_RESULTS[args.exp] + ".npz"))["test_mse"]
        res["original_test_mse"] = dict(bike=float(o[0]), nn=float(o[1]))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{args.exp}_s{args.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("TEST", json.dumps({k: res[k] for k in ["exp", "seed", "test_mse", "bike_params"]} | ({"original": res["original_test_mse"]} if "original_test_mse" in res else {})))


if __name__ == "__main__":
    main()
