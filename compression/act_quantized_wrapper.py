import torch
import compression
from torch import nn

from compression.eptq.eptq_config import EptqConfig, AnnealingFunction
from compression.quantization.quantizers import uniform_quantization, lsq_trainable_uniform_quantization
from constants import DEVICE, SIGMOID_MINUS


def calculate_uniform_params(act_nbits, min_val, max_val):
    bound_quant_min = 0
    bound_quant_max = 2 ** act_nbits - 1

    quant_min, quant_max = bound_quant_min, bound_quant_max
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
    scale = torch.max(scale, torch.tensor(1e-8, dtype=torch.float32))
    zero_point = quant_min - torch.round(min_val_neg / scale)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)

    return scale, zero_point


def search_activation_params(x: torch.Tensor, act_nbits: int, n_steps: int = 100, mse_p: float = 2.0,
                             init_min: torch.Tensor = None, init_max: torch.Tensor = None):
    """
    Uniform quantizer min/max selection
    """
    bound_quant_min = 0
    bound_quant_max = 2 ** act_nbits - 1

    #################
    # Init min/max
    #################
    x_min, x_max = torch.aminmax(x) if (init_min is None and init_max is None) else (init_min, init_max)
    xrange = x_max - x_min
    best_score = torch.zeros_like(x_min) + (1e+10)
    best_min = x_min.clone()
    best_max = x_max.clone()

    ###############
    # Optimize
    ###############
    # enumerate xrange
    for i in range(1, n_steps + 1):
        tmp_min = torch.zeros_like(x_min)
        tmp_max = xrange / n_steps * i
        tmp_delta = (tmp_max - tmp_min) / float(bound_quant_max - bound_quant_min)
        # enumerate zp
        # for zp in range(bound_quant_min, bound_quant_max + 1):
        for zp in range(bound_quant_min, bound_quant_max + 1):
            new_min = tmp_min - zp * tmp_delta
            new_max = tmp_max - zp * tmp_delta

            ################################
            # Calculate scale and zero point
            ################################
            scale, zero_point = calculate_uniform_params(act_nbits, new_min, new_max)

            x_q = uniform_quantization(
                x, scale.item(), int(zero_point.item()),
                bound_quant_min, bound_quant_max)
            score = (x_q - x).abs().pow(mse_p).mean()

            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(best_score, score)
    return best_min, best_max


class ActivationQuantizedWrapper(nn.Module):
    def __init__(self, in_op, in_cc: compression.CompressionConfig, in_econfig: EptqConfig, name: str):
        super().__init__()

        if in_op is not None:
            self.add_module("act_op", in_op)
        else:
            self.act_op = None
        self.cc = in_cc
        self.name = name

        self.a_n_bits = self.cc.a_n_bits
        self.a_qmin, self.a_qmax = None, None

        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float).to(DEVICE))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int).to(DEVICE))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))

        self.search_qparams = False

        # QDROP
        self.active_qdrop = False
        self.disable_qdrop = False

        self.econfig = in_econfig

    def set_layer_qparams_search(self, v):
        self.search_qparams = v

    def forward(self, x):
        if self.act_op is not None:
            x = self.act_op(x)
        if self.search_qparams:
            self.a_qmin, self.a_qmax = search_activation_params(x, self.a_n_bits,
                                                                init_min=self.a_qmin,
                                                                init_max=self.a_qmax,
                                                                n_steps=100)

            _scale, _zero_point = calculate_uniform_params(self.a_n_bits, self.a_qmin, self.a_qmax)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        elif self.cc.activation_quantization:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

            if self.active_qdrop:
                x_orig = x

            if self.cc.act_lsq:
                grad_factor = 1.0 / (x.numel() * (2 ** self.a_n_bits - 1)) ** 0.5
            else:
                grad_factor = 1.0

            x = lsq_trainable_uniform_quantization(x, self.a_n_bits, self.scale, self.zero_point.item(),
                                                   grad_factor=grad_factor)

            if self.active_qdrop and not self.disable_qdrop:
                if self.econfig.qdrop_config.vanilla_qdrop:
                    drop_prob = self.econfig.qdrop_config.drop_start
                    x_prob = torch.where(torch.rand_like(x) < drop_prob, x_orig, x)
                else:
                    # annealing qdrop
                    t = self.econfig.get_eptq_iter()
                    if self.econfig.qdrop_config.annealing_fn == AnnealingFunction.LINEAR:
                        drop_prob = (((self.econfig.qdrop_config.drop_end - self.econfig.qdrop_config.drop_start) * t) /
                                     self.econfig.n_iters) + self.econfig.qdrop_config.drop_start
                    elif self.econfig.qdrop_config.annealing_fn == AnnealingFunction.COSINE:
                        # Assuming start and stop are 1 and 0 respectively
                        assert self.econfig.qdrop_config.drop_start == 1 and self.econfig.qdrop_config.drop_end == 0
                        drop_prob = torch.cos(torch.tensor(torch.pi / 2) * self.econfig.zero_one_intervals[t]).item()
                    elif self.econfig.qdrop_config.annealing_fn == AnnealingFunction.LOG:
                        # Assuming start and stop are 1 and 0 respectively
                        assert self.econfig.qdrop_config.drop_start == 1 and self.econfig.qdrop_config.drop_end == 0
                        drop_prob = -torch.log10(torch.tensor(0.9 * self.econfig.zero_one_intervals[t] + 0.1)).item()
                    else:
                        raise Exception()

                    if self.econfig.qdrop_config.grad_act_quant:
                        x_prob = drop_prob * x_orig + (1 - drop_prob) * x
                    else:
                        x_prob = torch.where(torch.rand_like(x) < drop_prob, x_orig, x)

                return x_prob

        return x

    def scaled_sigmoid_prob(self):
        return torch.sigmoid(self.learnable_prob - SIGMOID_MINUS)
