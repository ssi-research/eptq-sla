import copy
import torch
from torch import nn, fx
from torch.fx.experimental.optimization import fuse, matches_module_pattern, replace_node_module
from compression.quantized_wrapper import QuantizedWrapper
from compression.compression_config import CompressionConfig
from tqdm import tqdm


def replace2quantized_model(in_model: torch.nn.Module, in_cc: CompressionConfig,
                            inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(nn.Conv1d,),
                (nn.Conv2d,),
                (nn.Conv3d,)]
    if not inplace:
        in_model = copy.deepcopy(in_model)
    fx_model = fx.symbolic_trace(in_model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                replace_node_module(node.args[0], modules, QuantizedWrapper(conv, in_cc))

    return fx.GraphModule(fx_model, new_graph)


class QuantizationModelWrapper:
    def __init__(self, in_model, in_cc: CompressionConfig):
        print("Start BN Start")
        model_fold = fuse(in_model)
        print("End BN Fuse")
        print("Starting Layer Wrapping")
        self.qmodel = replace2quantized_model(model_fold, in_cc)
        print("End Layer Wrapping")

    def __call__(self, x):
        return self.qmodel(x)

    def apply_quantization(self):
        print("Apply Quantization")
        for n, m in tqdm(self.qmodel.named_modules()):
            if isinstance(m, QuantizedWrapper):
                m.apply_quantization()
