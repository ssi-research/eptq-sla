from enum import Enum


class ThresholdMethod(Enum):
    MSE = 0
    HMSE = 1


class CompressionConfig:
    def __init__(self, w_n_bits=8,
                 threshold_method: ThresholdMethod = ThresholdMethod.MSE):
        self.w_n_bits = w_n_bits
        self.threshold_method = threshold_method
        self.weights_quantization = False

    def enable_weights_quantization(self):
        self.weights_quantization = True
