"""우리 사본(models/ladder/ladder.py)이 경사를 끄면(use_grade=False) 팀원 원본과 출력이 같은지 확인한다.

  python models/ladder/tests/check_same_as_team.py <팀원 ladder_models 폴더>

같은 시드로 R1~R6을 만들고, 같은 상태(state_dict)를 넣은 뒤 무작위 입력에서 1스텝 출력과 25스텝 굴리기를 비교한다.
pedal 설정(AV21_IAC)과 force 설정(GTI_THUNDERHILL) 둘 다 본다. 대조로 경사를 켠 사본은 출력이 달라야 한다.
"""
import importlib.util, os, sys
from dataclasses import replace
import torch

OURS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, folder):
    spec = importlib.util.spec_from_file_location(name, os.path.join(folder, "ladder.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    team_dir = sys.argv[1]
    ours, team = load("ladder_ours", OURS), load("ladder_team", team_dir)
    sys.path.insert(0, team_dir)
    import configs as team_cfgs   # 팀원 configs (팀원 ladder 의 VehicleConfig) — 우리 쪽은 같은 값으로 다시 만든다
    torch.manual_seed(0)
    worst = 0.0
    for cname in ("AV21_IAC", "GTI_THUNDERHILL"):
        tc = team_cfgs.ALL[cname]
        oc = ours.VehicleConfig(**{k: getattr(tc, k) for k in tc.__dataclass_fields__})
        X = torch.randn(512, tc.hist, 6) * torch.tensor([10, 1, 0.3, 0.5 if tc.lon_input == "pedal" else 3000, 0.1, 0.02]) \
            + torch.tensor([25, 0, 0, 0, 0, 0])
        Y = X[:, -1:, :3] + 0.1 * torch.randn(512, 1, 3)
        U = X[:, -1:, 3:].repeat(1, 24, 1)
        for rung in ("R1", "R2", "R3", "R4", "R5", "R6"):
            torch.manual_seed(1); mt = team.build(rung, tc, team.Norm(X, Y))
            torch.manual_seed(1); mo = ours.build(rung, oc, ours.Norm(X, Y))
            mo.load_state_dict(mt.state_dict())
            with torch.no_grad():
                d1 = (mt(X) - mo(X)).abs().max().item()
                d25 = (team.rollout(mt, X, U) - ours.rollout(mo, X, U)).abs().max().item()
            worst = max(worst, d1, d25)
            print(f"{cname:16s} {rung}: 1스텝 최대 차 {d1:.2e}, 25스텝 최대 차 {d25:.2e}")
        # 대조: 경사를 켜고 경사 채널에 0이 아닌 값을 넣으면 달라져야 한다
        og = replace(oc, use_grade=True)
        Xg = torch.cat([X, torch.full((512, tc.hist, 1), 0.05)], -1)
        torch.manual_seed(1); mg = ours.build("R1", og, ours.Norm(Xg, Y))
        torch.manual_seed(1); m1 = team.build("R1", tc, team.Norm(X, Y))
        mg.g.raw.data.copy_(m1.g.raw.data)          # R1은 물리 계수만 있다 (표준화 버퍼는 채널 수가 달라 복사하지 않음)
        with torch.no_grad():
            dg = (m1(X)[:, 0] - mg(Xg)[:, 0]).abs().mean().item()
        print(f"{cname:16s} 대조(경사 0.05 rad 켜기): vx 평균 차 {dg:.4f} m/s (0.04 s × g·sin(0.05) ≈ {0.04 * 9.81 * 0.05:.4f} 기대)")
    print("통과" if worst == 0.0 else f"차이 있음 (최대 {worst:.2e})")
    sys.exit(0 if worst == 0.0 else 1)


if __name__ == "__main__":
    main()
