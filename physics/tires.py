"""타이어 횡력 모델과 마찰원 결합. 부호 규약: 슬립각이 양수면 횡력은 음수(Spielberg 코드와 같음)."""
import math
import torch


def linear_lat(alpha, fz, mu, tire):
    """P0: 선형 타이어."""
    return -tire.c_alpha * fz * alpha


def fiala_lat(alpha, fz, mu, tire):
    """P1 대안: Fiala(브러시) 모델. Spielberg models.py의 fiala_tf와 같은 식(tan 사용)."""
    Ca = tire.c_alpha * fz
    ta = torch.tan(alpha)
    alpha_slide = torch.abs(torch.atan(3 * mu * fz / Ca))
    lin = (-Ca * ta + Ca ** 2 / (3 * mu * fz) * torch.abs(ta) * ta
           - Ca ** 3 / (27 * mu ** 2 * fz ** 2) * ta ** 3)
    return torch.where(torch.abs(alpha) < alpha_slide, lin, -mu * fz * torch.sign(alpha))


def pacejka_lat(alpha, fz, mu, tire):
    """P1: Pacejka magic formula. D = mu*Fz, B는 코너링 강성이 c_alpha*Fz가 되도록 정한다."""
    B = tire.c_alpha / (tire.C * mu)
    x = B * alpha
    return -mu * fz * torch.sin(tire.C * torch.atan(x - tire.E * (x - torch.atan(x))))


TIRE_MODELS = {"linear": linear_lat, "fiala": fiala_lat, "pacejka": pacejka_lat}


def peak_slip(tire, mu, model="pacejka"):
    """횡력이 최대가 되는 슬립각 [rad]. 구간 라벨(|alpha|/alpha_peak)에 쓴다."""
    if model == "pacejka":
        if tire.E != 0.0:
            raise NotImplementedError("E != 0 이면 수치 탐색 필요")
        return math.tan(math.pi / (2 * tire.C)) * tire.C * mu / tire.c_alpha
    if model == "fiala":
        return torch.atan(torch.as_tensor(3 * mu / tire.c_alpha, dtype=torch.float64))
    raise ValueError("선형 타이어는 포화가 없다")


def derate(fx, fz, mu, floor=0.02):
    """종방향 힘을 μFz·sqrt(1-floor²) 안으로 자르고, 남은 횡방향 최대 마찰계수를 돌려준다.
    횡력 모델에 이 μ를 넣으면 sqrt(fx^2 + fy^2) <= mu*Fz 가 항상 성립한다.
    floor는 횡방향 μ가 0이 되어 Pacejka B가 무한대가 되는 것을 막는다."""
    fmax = mu * fz * math.sqrt(1.0 - floor ** 2)
    fx_c = torch.maximum(torch.minimum(fx, fmax), -fmax)
    mu_y = torch.sqrt(torch.clamp(mu ** 2 - (fx_c / fz) ** 2, min=(floor * mu) ** 2))
    return fx_c, mu_y


def friction_ellipse(fx, fy_pure, fz, mu, eps=1e-9):
    """P2: 종방향 힘을 μFz 안으로 자르고, 남은 여유만큼 횡력을 줄인다.
    sqrt(fx^2 + fy^2) <= mu*Fz 가 항상 성립한다."""
    fmax = mu * fz
    fx_c = torch.maximum(torch.minimum(fx, fmax), -fmax)
    scale = torch.sqrt(torch.clamp(1.0 - (fx_c / fmax) ** 2, min=0.0) + eps)
    return fx_c, fy_pure * torch.clamp(scale, max=1.0)
