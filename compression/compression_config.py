from enum import Enum
from typing import Tuple, Any
import numpy as np


class ThresholdMethod(Enum):
    MSE = 0
    HMSE = 1


class CompressionConfig:
    def __init__(self,
                 w_n_bits: int = 8,
                 a_n_bits: int = 8,
                 enable_act_quantization: bool = False,
                 threshold_method: ThresholdMethod = ThresholdMethod.HMSE,
                 mse_p: float = 2.0,
                 qparam_train: bool = False,
                 first_last_eight_bit: bool = False,
                 act_lsq: bool = False,
                 qdrop: bool = False):

        # Weights
        self.w_n_bits = w_n_bits
        self.threshold_method = threshold_method
        self.weights_quantization = False

        # Activation
        self.enable_act_quantization = enable_act_quantization
        self.a_n_bits = a_n_bits
        self.activation_quantization = False
        self.act_lsq = act_lsq

        # EPTQ
        self.qparam_train = qparam_train
        self.qdrop = qdrop

        # Config
        self.mse_p = mse_p
        self.first_last_eight_bit = first_last_eight_bit

    def set_weights_quantization(self, quantize_weights: bool):
        self.weights_quantization = quantize_weights

    def set_act_quantization(self, quantize_activation: bool):
        self.activation_quantization = quantize_activation

    def update_weights_bit_width(self, w_n_bits):
        self.w_n_bits = w_n_bits

    def update_act_bit_width(self, a_n_bits):
        self.a_n_bits = a_n_bits
