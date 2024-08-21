from typing import Callable

import numpy as np
import torch
from tqdm import tqdm
from compression.quantized_wrapper import QuantizedWrapper
import constants as C
from torch import autograd


class HessianConfig:
    def __init__(self, n_iter: int = 100, min_iterations: int = 10, threshold: float = 1e-4,
                 criterion=torch.nn.CrossEntropyLoss()):

        self.n_iter = n_iter
        self.min_iterations = min_iterations
        self.threshold = threshold
        self.debug_plot = False
        self.criterion = criterion


def weight_lfh(in_images, in_model, hconfig: HessianConfig = HessianConfig()):
    print("Start Weight Label Free Hessian")
    ##############################
    # Compute
    ##############################
    results_dict = {n: [] for n, m in in_model.named_modules() if isinstance(m, QuantizedWrapper)}

    for image_index in range(in_images.shape[0]):
        print(f"\nImage {image_index + 1} / {in_images.shape[0]}")
        image_results_dict = {n: 0 for n, m in in_model.named_modules() if isinstance(m, QuantizedWrapper)}
        _x = in_images[image_index, :].unsqueeze(dim=0)
        y_hat = in_model(_x.to(C.DEVICE))

        re_dict = {n: np.inf for n, m in in_model.named_modules() if isinstance(m, QuantizedWrapper)}

        for i in tqdm(range(hconfig.n_iter)):
            v = torch.randn_like(y_hat)
            l = torch.mean(y_hat.unsqueeze(dim=1) @ v.unsqueeze(dim=-1))
            for n, m in in_model.named_modules():
                if isinstance(m, QuantizedWrapper):
                    jac_v = autograd.grad(outputs=l,
                                          inputs=m.op.weight,  # Change 1: take derivative w.r.t to weights
                                          retain_graph=True)[0]

                    lfh = torch.pow(jac_v, 2.0).detach().cpu().numpy()

                    new_value = (i * image_results_dict[n] + lfh) / (i + 1)
                    if i > hconfig.min_iterations:
                        re_dict[n] = np.max(np.abs(new_value - image_results_dict[n]) / (np.abs(new_value) + 1e-6))
                    image_results_dict[n] = new_value
            if i > hconfig.min_iterations and np.all(np.asarray([v for v in re_dict.values()]) < hconfig.threshold):
                break

        for k, v in image_results_dict.items():
            results_dict[k].append(v)

    return {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in results_dict.items()}

