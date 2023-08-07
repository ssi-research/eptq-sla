import torch
from torch import nn

import compression


def quantization(in_w, in_n_bits, in_threshold):
    q_points = 2 ** (in_n_bits - 1)
    delta_array = in_threshold / q_points
    return delta_array * torch.clamp(torch.round(in_w / delta_array), -q_points, q_points - 1)


def search_threshold(in_w, in_n_bits, channel_axis, n_iter=3, n_steps=100, sweep_scale=0.2, min_scale=0.05):
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
            q_w = quantization(w_tilde, in_n_bits, threshold_array)
            mse = torch.linalg.norm(w_tilde - q_w, dim=-1, ord=2) / w_reshape.shape[-1]
            index = mse.argmin(dim=0).flatten()
            threshold = scale_array.flatten()[index] * threshold.flatten()
            threshold = threshold.reshape([1, -1, 1])

            ########################################
            # Update for the next iter
            #######################################
            eps = (sweep_scale - min_scale) * (i + 1) / n_iter
            scale_array = torch.linspace(1 - sweep_scale + eps, 1 + sweep_scale - eps, steps=n_steps,
                                         device=in_w.device).reshape([-1, 1, 1, 1])

    return threshold.flatten().detach()


class QuantizedWrapper(nn.Module):
    def __init__(self, in_float_conv, in_cc: compression.CompressionConfig):
        super().__init__()
        if not isinstance(in_float_conv, nn.Conv2d):
            raise Exception("A")
        self.add_module("conv", in_float_conv)
        self.cc = in_cc

        self.original_w = nn.Parameter(in_float_conv.weight.detach().clone(), requires_grad=True)
        self.original_bias = nn.Parameter(in_float_conv.bias.detach().clone(), requires_grad=True)
        
        self.quantized_w = None
        self.quantized_bias = None
        self.threshold = None

    def apply_quantization(self):
        self.threshold = search_threshold(self.original_w.clone(), self.cc.w_n_bits, 0)
        self.quantized_w = quantization(self.original_w.clone(), self.cc.w_n_bits,
                                        self.threshold.reshape([-1, 1, 1, 1]))

    def forward(self, x):
        if self.cc.weights_quantization:
            self.conv.weight.data = self.quantized_w
            self.conv.bias = self.original_bias
        else:
            self.conv.weight = self.original_w
            self.conv.bias = self.original_bias
        return self.conv(x)
