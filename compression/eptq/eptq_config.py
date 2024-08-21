from enum import Enum
import numpy as np


class AnnealingFunction(Enum):
    LINEAR = 1,
    COSINE = 2,
    LOG = 3


class QdropConfig:
    def __init__(self,
                 drop_start: float = 1.0,
                 drop_end: float = 0.0,
                 grad_act_quant: bool = False,
                 vanilla_qdrop: bool = False,
                 annealing_fn: AnnealingFunction = AnnealingFunction.LINEAR):

        # Qdrop params
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.grad_act_quant = grad_act_quant
        self.vanilla_qdrop = vanilla_qdrop
        self.annealing_fn = annealing_fn


class EptqConfig:

    def __init__(self, lr: float = 3e-2, n_iters: int = 20000, bias_train: bool = False, lr_bias: float = 1e-4,
                 qparam_train: bool = False, lr_qparam: float = 1e-3, num_samples: int = 1024,
                 reg_factor: float = 1e-2, act_scale_lr: float = 4e-5,
                 b_decay_mode: str = 'linear', qdrop_config: QdropConfig = QdropConfig()):

        self.lr = lr
        self.n_iters = n_iters
        self.bias_train = bias_train
        self.lr_bias = lr_bias
        self.qparam_train = qparam_train
        self.lr_qparam = lr_qparam
        self.num_samples = num_samples
        self.reg_factor = reg_factor
        self.act_scale_lr = act_scale_lr
        self.qdrop_config = qdrop_config

        assert b_decay_mode in ['linear', 'cosine']
        self.b_decay_mode = b_decay_mode

        self.t = 0

        self.zero_one_intervals = list(np.linspace(0, 1, n_iters))

    def increment_eptq_iter(self):
        self.t += 1

    def get_eptq_iter(self):
        return self.t


