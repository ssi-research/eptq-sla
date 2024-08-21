from typing import List

import torch
import math

from compression import CompressionConfig
from compression.act_quantized_wrapper import ActivationQuantizedWrapper
from compression.eptq.eptq_config import EptqConfig
from compression.quantization.quantizers import get_soft_targets
from compression.quantized_wrapper import QuantizedWrapper


class CosineTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            t1, t2 = t - self.start_decay, self.t_max - self.start_decay
            return self.end_b + (self.start_b - self.end_b) * (1 + math.cos(math.pi * t1 / t2)) / 2


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LFHLossFunctionHessianPerImage:
    def __init__(self,
                 trainable_layers: List[QuantizedWrapper],
                 act_trainable_layers: List[ActivationQuantizedWrapper],
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 cc: CompressionConfig = None,
                 econfig: EptqConfig = None):

        self.reg_factor = econfig.reg_factor
        self.loss_start = econfig.n_iters * warmup
        self.p = p

        self.count = 0

        if econfig.b_decay_mode == 'linear':
            temp_decay_object = LinearTempDecay
        elif econfig.b_decay_mode == 'cosine':
            temp_decay_object = CosineTempDecay
        else:
            raise NotImplemented
        self.temp_decay = temp_decay_object(econfig.n_iters, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                            start_b=b_range[0], end_b=b_range[1])

        self.trainable_layers = trainable_layers
        self.act_trainable_layers = act_trainable_layers

        self.q_activation_tensors = []
        self.f_activation_tensors = []

        self.cc = cc
        self.econfig = econfig

    def __call__(self, hessian):
        s = []
        b = self.temp_decay.start_b if self.count < self.loss_start else self.temp_decay(self.count)

        round_loss = 0
        max_w = []
        round_loss_base_sum = 0

        for i, layer in enumerate(self.trainable_layers):
            # Reconstruction loss
            q_tensor = [a[1] for a in self.q_activation_tensors if a[0] == layer.name][0]
            f_tensor = [a[1] for a in self.f_activation_tensors if a[0] == layer.name][0]

            w = hessian[layer.name]

            norm = (q_tensor - f_tensor).pow(self.p).sum(1)
            if len(norm.shape) > 1:
                norm = norm.flatten(1).mean(1)
            x = torch.mean(w * norm)
            max_w.append(torch.mean(w))
            s.append(x)

            # Round loss
            round_vals = get_soft_targets(layer.alpha, layer.gamma, layer.zeta)
            round_loss_base = (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            round_loss += torch.mean(w) * self.reg_factor * round_loss_base
            round_loss_base_sum += round_loss_base

        rec_loss = torch.sum(torch.stack(s))
        total_loss = rec_loss + round_loss

        # clear activation tensors structures
        self.q_activation_tensors.clear()
        self.f_activation_tensors.clear()
        s.clear()

        self.count += 1

        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))

        max_w = torch.max(torch.stack(max_w))
        total_loss = total_loss / max_w

        return total_loss, rec_loss, float(round_loss_base_sum), float(round_loss), b
