"""미분 가능한 자전거(single-track) 모델. P0~P2를 옵션으로 고른다.

상태 x = [X, Y, psi, Ux, Uy, r]  (전역 위치·방향은 궤적 시각화용)
입력 u = [delta, Fx]            (Fx: 바퀴에서의 전체 종방향 힘 명령 [N], 양수 구동 / 음수 제동)

  P0: tire="linear",  load_transfer=False, combined=False
  P1: tire="pacejka", load_transfer=False, combined=False   (또는 tire="fiala")
  P2: tire="pacejka", load_transfer=True,  combined="derate"  (기본값)

결합 슬립(combined): "derate"는 종방향 힘이 쓴 만큼 횡방향 최대 마찰만 줄이고 코너링 강성은 유지한다.
True/"ellipse"는 횡력 곡선 전체를 줄이는 마찰 타원(11/7 이전 기본값). CommonRoad 대조에서 선회 중
가속 거동이 derate 쪽이 훨씬 가까워 기본값을 바꿨다(docs/commonroad_check_2026-11-07.md).

하중 이동은 Fx/m을 종가속도로 보는 준정적 근사다(Spielberg data_generation_functions.py와 같음).
"""
from dataclasses import dataclass
import torch

from .tires import TIRE_MODELS, friction_ellipse, derate

STATE = ("X", "Y", "psi", "ux", "uy", "r")
INPUT = ("delta", "fx")


@dataclass
class ModelOptions:
    tire: str = "pacejka"
    load_transfer: bool = True
    combined: object = "derate"   # False | "derate" | True(="ellipse")
    resistance: bool = True


P0 = ModelOptions(tire="linear", load_transfer=False, combined=False)
P1 = ModelOptions(tire="pacejka", load_transfer=False, combined=False)
P2 = ModelOptions()


def forces(x, u, p, mu, opt=P2):
    """타이어 힘, 수직 하중, 슬립각."""
    ux, uy, r = x[..., 3], x[..., 4], x[..., 5]
    delta, fx = u[..., 0], u[..., 1]
    m, g, a, b, L = p.m, p.g, p.a, p.b, p.L

    fxf = torch.where(fx > 0, p.drive_front * fx, p.brake_front * fx)
    fxr = fx - fxf

    if opt.load_transfer:
        ax = fx / m
        fzf = m * g * b / L - p.h * m * ax / L
        fzr = m * g * a / L + p.h * m * ax / L
        fzf = torch.clamp(fzf, min=0.05 * m * g)
        fzr = torch.clamp(fzr, min=0.05 * m * g)
    else:
        fzf = m * g * b / L * torch.ones_like(ux)
        fzr = m * g * a / L * torch.ones_like(ux)

    alpha_f = torch.atan((uy + a * r) / ux) - delta
    alpha_r = torch.atan((uy - b * r) / ux)
    tire = TIRE_MODELS[opt.tire]
    if opt.combined == "derate":
        # 종방향 힘이 쓴 만큼 횡방향 최대 마찰만 줄인다: μ_y = sqrt(μ² - (Fx/Fz)²).
        # 코너링 강성(c_alpha·Fz)은 그대로라서 선형 구간은 거의 안 바뀐다.
        fxf, muf = derate(fxf, fzf, mu)
        fxr, mur = derate(fxr, fzr, mu)
        fyf = tire(alpha_f, fzf, muf, p.tire_f)
        fyr = tire(alpha_r, fzr, mur, p.tire_r)
    else:
        fyf = tire(alpha_f, fzf, mu, p.tire_f)
        fyr = tire(alpha_r, fzr, mu, p.tire_r)
        if opt.combined:   # True 또는 "ellipse": 횡력 곡선 전체를 sqrt(1-(Fx/μFz)²)배
            fxf, fyf = friction_ellipse(fxf, fyf, fzf, mu)
            fxr, fyr = friction_ellipse(fxr, fyr, fzr, mu)

    return dict(fxf=fxf, fxr=fxr, fyf=fyf, fyr=fyr, fzf=fzf, fzr=fzr, alpha_f=alpha_f, alpha_r=alpha_r)


def derivatives(x, u, p, mu, opt=P2, return_forces=False, slope=None):
    """slope: (..., 2) [경사 rad (오르막 +), 좌우 기울기 rad (차 왼쪽(+y)으로 중력이 끄는 방향 +)] 또는 None(평지).
    중력 성분만 더한다: dUx/dt −= g sin(경사), dUy/dt += g sin(기울기). 수직 하중 변화(cos 항)는 4°에서 0.3%라 무시."""
    psi, ux, uy, r = x[..., 2], x[..., 3], x[..., 4], x[..., 5]
    delta = u[..., 0]
    f = forces(x, u, p, mu, opt)
    cd, sd = torch.cos(delta), torch.sin(delta)
    f_res = (p.c0 + p.c1 * ux + p.c2 * ux ** 2) if opt.resistance else 0.0

    ux_dot = (f["fxf"] * cd - f["fyf"] * sd + f["fxr"] - f_res) / p.m + r * uy
    uy_dot = (f["fyf"] * cd + f["fxf"] * sd + f["fyr"]) / p.m - r * ux
    r_dot = (p.a * (f["fyf"] * cd + f["fxf"] * sd) - p.b * f["fyr"]) / p.Iz
    if slope is not None:
        ux_dot = ux_dot - p.g * torch.sin(slope[..., 0])
        uy_dot = uy_dot + p.g * torch.sin(slope[..., 1])
    X_dot = ux * torch.cos(psi) - uy * torch.sin(psi)
    Y_dot = ux * torch.sin(psi) + uy * torch.cos(psi)

    xd = torch.stack([X_dot, Y_dot, r, ux_dot, uy_dot, r_dot], -1)
    if return_forces:
        f["ax"] = ux_dot - r * uy   # 차체 좌표계 가속도 (IMU가 재는 값)
        f["ay"] = uy_dot + r * ux
        return xd, f
    return xd


def rk4_step(x, u, p, mu, dt, opt=P2, slope=None):
    """입력(과 노면 경사·기울기)을 dt 동안 고정(ZOH)하고 RK4 한 스텝."""
    k1 = derivatives(x, u, p, mu, opt, slope=slope)
    k2 = derivatives(x + 0.5 * dt * k1, u, p, mu, opt, slope=slope)
    k3 = derivatives(x + 0.5 * dt * k2, u, p, mu, opt, slope=slope)
    k4 = derivatives(x + dt * k3, u, p, mu, opt, slope=slope)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(x, u, p, mu, dt, opt=P2):
    return x + dt * derivatives(x, u, p, mu, opt)
