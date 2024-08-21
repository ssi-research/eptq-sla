import os
import pickle
from typing import Callable

import numpy as np
import torch
from torch import autograd
from tqdm import tqdm

import constants as C
from compression.hessian.hooks import model_register_hook
from compression.hessian.lfh import HessianConfig
from compression.quantized_wrapper import QuantizedWrapper
from torch.utils.data import Dataset


class HessianDataset(Dataset):
    def __init__(self, base_dataset, hessian_data):
        self.data = base_dataset
        self.hessian_data = hessian_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.__getitem__(index)
        return data[0], {k: v[index] for k, v in self.hessian_data.items()}


def generate_hessian_scale_dataset(in_qmw, in_dataloader, layers_types=QuantizedWrapper,
                                   hconfig: HessianConfig = HessianConfig()):
    print(f"Computing Activation LFH...")
    activations = {}
    hook_handles = []
    activation_quantization = in_qmw.cc.activation_quantization
    weights_quantization = in_qmw.cc.weights_quantization
    in_qmw.cc.set_act_quantization(False)
    in_qmw.cc.set_weights_quantization(False)
    model_register_hook(in_qmw.qmodel, activations, hook_handles, layers_types)
    non_random_dataloader = torch.utils.data.DataLoader(in_dataloader.dataset, in_dataloader.batch_size, shuffle=False,
                                                        num_workers=1)
    final_scores_per_layer_per_image = None
    for x in non_random_dataloader:
        activations.clear()
        _final_scores_per_layer_per_image = activation_lfh_max_fast_batch_iter(x[0], in_qmw.qmodel, layers_types,
                                                                               activations,
                                                                               hconfig)
        if final_scores_per_layer_per_image is None:
            final_scores_per_layer_per_image = _final_scores_per_layer_per_image
        else:
            final_scores_per_layer_per_image = {k: np.concatenate([final_scores_per_layer_per_image[k], v], axis=0) for
                                                k, v in
                                                _final_scores_per_layer_per_image.items()}
    for handle in hook_handles:
        handle.remove()

    in_qmw.cc.set_act_quantization(activation_quantization)
    in_qmw.cc.set_weights_quantization(weights_quantization)
    hd = HessianDataset(in_dataloader.dataset, final_scores_per_layer_per_image)
    return torch.utils.data.DataLoader(hd, in_dataloader.batch_size, shuffle=True,
                                       num_workers=in_dataloader.num_workers, pin_memory=True)


def activation_lfh_max_fast_batch_iter(in_images, in_model, layers_types, activations, hconfig):
    _x = in_images
    output = in_model(_x.to(C.DEVICE))
    per_layer_image_results_dict = {m.name: 0 for n, m in in_model.named_modules() if
                                    isinstance(m, layers_types)}
    for i in tqdm(range(hconfig.n_iter)):  # iterations over random vectors
        v = torch.randint_like(output, high=2, device=C.DEVICE)
        v[v == 0] = -1
        out_v = torch.sum(v * output)
        for name, activation_tensor in activations.items():  # for each layer's output
            jac_v = autograd.grad(outputs=out_v,
                                  inputs=activation_tensor,
                                  retain_graph=True)[0]
            with torch.no_grad():
                # Tr(H) ~= Sum(J^2)
                if len(jac_v.shape) == 4:
                    per_layer_lfh = torch.mean(jac_v ** 2, dim=(2, 3))
                else:
                    per_layer_lfh = jac_v ** 2
                per_layer_lfh = per_layer_lfh.detach()  # Batch, Channel
                # E(Tr(H)) - E is over the random iterations
                per_layer_lfh_new_val = (i * per_layer_image_results_dict[name] + per_layer_lfh) / (i + 1)

                per_layer_image_results_dict[name] = per_layer_lfh_new_val
    final_scores_per_layer_per_image = {k: np.max(v.detach().cpu().numpy(), axis=1) for k, v in
                                        per_layer_image_results_dict.items()}  # Max Per Tensor
    return final_scores_per_layer_per_image
