import torch
import numpy as np
import compression
from torch import nn

from compression.compression_config import ThresholdMethod
from compression.quantization.quantizers import symmetric_quantization, soft_quantization


def search_weights_threshold(in_w, in_n_bits, channel_axis, n_iter=15, n_steps=100, sweep_scale=0.2, min_scale=0.05,
                             hessian=None, mse_p=2.0):
    with torch.no_grad():
        ########################
        # Reorder
        ########################
        order_array = [channel_axis, *[i for i in range(len(in_w.shape)) if i != channel_axis]]
        w_perm = torch.permute(in_w, order_array)
        n_channels = w_perm.shape[0]
        w_reshape = w_perm.reshape([1, n_channels, -1])
        ########################
        # Threshold Loop
        ########################
        threshold = (1 + sweep_scale) * torch.max(torch.abs(w_reshape), dim=-1)[0]
        threshold = threshold.unsqueeze(dim=-1)
        scale_array = torch.linspace(1 - sweep_scale, 1 + sweep_scale, steps=n_steps, device=in_w.device).reshape(
            [-1, 1, 1, 1])
        for i in range(n_iter):
            threshold_array = scale_array * threshold

            w_tilde = w_reshape.unsqueeze(dim=0)
            q_w = symmetric_quantization(w_tilde, in_n_bits, threshold_array)
            if hessian is None:
                mse = torch.linalg.norm(w_tilde - q_w, dim=-1, ord=mse_p) ** 2 / w_reshape.shape[-1]
            else:
                hessian_perm = torch.permute(hessian, order_array)
                hessian_reshape = hessian_perm.reshape([1, 1, n_channels, -1])
                mse = torch.linalg.norm(hessian_reshape * (w_tilde - q_w), dim=-1, ord=mse_p) ** 2 / w_reshape.shape[-1]
            index = mse.argmin(dim=0).flatten()
            threshold = scale_array.flatten()[index] * threshold.flatten()
            threshold = threshold.reshape([1, -1, 1])

            ########################################
            # Update for the next iter
            #######################################
            eps = (sweep_scale - min_scale) * (i + 1) / n_iter
            scale_array = torch.linspace(1 - sweep_scale + eps, 1 + sweep_scale - eps, steps=n_steps + 1,
                                         device=in_w.device).reshape([-1, 1, 1, 1])

    return threshold.flatten().detach()


class QuantizedWrapper(nn.Module):
    def __init__(self, in_op, in_cc: compression.CompressionConfig, name: str):
        super().__init__()

        if not (isinstance(in_op, nn.Conv2d) or isinstance(in_op, nn.Linear)):
            raise Exception(f"Unknown Operations Type{type(in_op)}")

        # Module
        self.add_module("op", in_op)
        self.cc = in_cc
        self.is_conv = isinstance(in_op, nn.Conv2d)
        self.name = name

        # Params
        self.original_w = nn.Parameter(in_op.weight.detach().clone(), requires_grad=True)
        self.original_bias = nn.Parameter(in_op.bias.detach().clone(), requires_grad=True)

        self.w = nn.Parameter(in_op.weight.detach().clone(), requires_grad=True)
        self.bias = nn.Parameter(in_op.bias.detach().clone(), requires_grad=True)

        # Quantization
        self.w_n_bits = self.cc.w_n_bits
        self.w_threshold = None

        # Soft quantizer parameters and EPTQ
        self.reconstructed = False
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.alpha = None
        self.qparam_train = in_cc.qparam_train

    def _safe_log(self, x, eps):
        return torch.log(torch.max(x, torch.Tensor([eps]).to(x.device)))

    def init_alpha(self, x: torch.Tensor, eps=1e-8):
        q_points = 2 ** (self.w_n_bits - 1)
        delta = self.w_threshold / q_points
        x_floor = torch.floor(x / (delta+ eps))
        rest = (x / (delta + eps)) - x_floor  # rest of rounding [0, 1)
        alpha = -self._safe_log((self.zeta - self.gamma) / (rest - self.gamma) - 1, 1e-16)  # => sigmoid(alpha) = rest
        self.alpha = nn.Parameter(alpha)

    def init_module_reconstruction(self):
        self.init_alpha(self.original_w.clone())
        if self.qparam_train:
            self.init_threshold_scale()
        self.reconstructed = True

    def init_threshold_scale(self):
        self.register_parameter(f"{self.name.replace('.', '_')}_scale",
                                nn.Parameter(torch.ones_like(torch.Tensor(self.w_threshold)), requires_grad=True))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))

    def add_hessian_information(self, hessian):
        self.hessian = hessian

    def _set_weights_threshold(self):
        threshold = search_weights_threshold(self.original_w.clone(), self.w_n_bits, 0,
                                             hessian=torch.Tensor(np.sqrt(self.hessian)).to(
                                                 self.original_w.device) if self.cc.threshold_method == ThresholdMethod.HMSE
                                             else None,
                                             mse_p=self.cc.mse_p)
        self.w_threshold = threshold.reshape([-1, 1, 1, 1] if self.is_conv else [-1, 1])

    def apply_weights_quantization(self):
        self._set_weights_threshold()

    def forward(self, x):
        del self.op.weight
        del self.op.bias
        if self.cc.weights_quantization:
            if self.reconstructed:
                self.op.weight = soft_quantization(w=self.w, alpha=self.alpha, n_bits=self.w_n_bits,
                                                   threshold=self.w_threshold, gamma=self.gamma, zeta=self.zeta,
                                                   training=self.training)
                if self.qparam_train:
                    scale = self.get_parameter(f"{self.name.replace('.', '_')}_scale")
                    scale.data.abs_()
                    scale.data.clamp_(min=self.eps.item())
                    self.op.weight *= scale
            else:
                if self.simd < np.inf:
                    # If we are in SIMD mode then EPTQ is not supported and we load pre-saved weights instead
                    # of quantizing on the fly
                    self.op.weight = self.w
                else:
                    self.op.weight = symmetric_quantization(self.w, self.w_n_bits, self.w_threshold)

            self.op.bias = self.original_bias
        else:
            self.op.weight = self.original_w
            self.op.bias = self.original_bias

        return self.op(x)
