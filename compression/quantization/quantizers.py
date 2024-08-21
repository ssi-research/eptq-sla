import torch

#################
# Quantizers
#################


def symmetric_quantization(in_w, in_n_bits, in_threshold):
    q_points = 2 ** (in_n_bits - 1)
    delta_array = in_threshold / q_points
    return delta_array * torch.clamp(torch.round(in_w / delta_array), -q_points, q_points - 1)


def uniform_quantization(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def lsq_trainable_uniform_quantization(x, n_bits, scale, zero_point, grad_factor=0.0):
    quant_min = 0
    quant_max = 2 ** n_bits - 1
    scale = grad_scale(scale, grad_factor)

    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale

    return x_dequant


def soft_quantization(w, alpha, n_bits, threshold, gamma, zeta, training=False, eps=1e-8):
    q_points = 2 ** (n_bits - 1)
    delta = threshold / q_points
    w_floor = floor_ste(w / (delta + eps))

    if training:
        w_int = w_floor + get_soft_targets(alpha, gamma, zeta)
    else:
        w_int = w_floor + (alpha >= 0).float()

    w_quant = ste_clip(w_int, -q_points, q_points - 1)
    w_float_q = w_quant * delta

    return w_float_q


#################
# Helpers
#################

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def ste_clip(x: torch.Tensor, min_val=-1.0, max_val=1.0) -> torch.Tensor:
    """
    Clip a variable between fixed values such that min_val<=output<=max_val
    Args:
        x: input variable
        min_val: minimum value for clipping
        max_val: maximum value for clipping
    Returns:
        clipped variable
    """
    return (torch.clip(x, min=min_val, max=max_val) - x).detach() + x


def floor_ste(x):
    return (torch.floor(x) - x).detach() + x


def get_soft_targets(alpha, gamma, zeta):
    return torch.clamp(torch.sigmoid(alpha) * (zeta - gamma) + gamma, 0, 1)


def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)