import copy
from typing import Dict, Iterable, Type, Any, Tuple

import torch
from torch import nn, fx
from torch.fx.experimental.optimization import fuse

from compression.act_quantized_wrapper import ActivationQuantizedWrapper
from compression.eptq.eptq_config import EptqConfig
from compression.quantized_wrapper import QuantizedWrapper
from compression.compression_config import CompressionConfig
from tqdm import tqdm
from compression.hessian.lfh import weight_lfh, HessianConfig
from constants import LINEAR_OPS, ACTIVATION_OPS


def _is_last_linear(in_node, last_linear_layer):
    return last_linear_layer is not None and in_node == last_linear_layer


def replace2quantized_model(in_model: torch.nn.Module, in_cc: CompressionConfig, inplace=False,
                            linear_patterns=LINEAR_OPS, act_patterns=ACTIVATION_OPS, is_act_quantization=False,
                            last_linear_layer: str = None, in_econfig: EptqConfig = None) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """

    if not inplace:
        in_model = copy.deepcopy(in_model)
    else:
        in_model = in_model
    fx_model = fx.symbolic_trace(in_model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in linear_patterns:
        for node in new_graph.nodes:
            for in_node in node.args:
                if _matches_module_pattern(pattern, node, in_node, modules):
                    target_op = modules[in_node.target]

                    wrap_node = _replace_node_module(in_node, modules,
                                                     QuantizedWrapper(target_op, in_cc, name=in_node.target))
                    if is_act_quantization:
                        succs_nodes = [m for m in new_graph.nodes if in_node in m.args]
                        # (1) If this node is the last linear op then we are not quantizing its activation
                        # (2) if there is a conv -> relu in the graph then we only quantize the relu activation output,
                        #     otherwise, we wrap the convolution with a weights quantizer wrapper (QuantizedWrapper)
                        #     and activation quantizer wrapper (ActivationQuantizedWrapper) on top of it.
                        if not _is_last_linear(in_node.target, last_linear_layer):
                            if ('add' in in_node.next.name or isinstance(modules[in_node.next.target], (nn.ReLU, nn.ReLU6))
                                    or 'downsample' in in_node.next.name) \
                                    or any(['add' in s.name or isinstance(modules[s.target], (nn.ReLU, nn.ReLU6))
                                    or 'downsample' in s.name for s in succs_nodes]):
                                continue
                            else:
                                _replace_node_module(in_node, modules,
                                                     ActivationQuantizedWrapper(wrap_node, in_cc, name=in_node.target,
                                                                                in_econfig=in_econfig))
    if is_act_quantization:
        for pattern in act_patterns:
            for node in new_graph.nodes:
                for in_node in node.args:
                    if _matches_module_pattern(pattern, node, in_node, modules):
                        target_op = modules[in_node.target]
                        _replace_node_module(in_node, modules, ActivationQuantizedWrapper(target_op, in_cc,
                                                                                          name=in_node.target,
                                                                                          in_econfig=in_econfig))

    return fx.GraphModule(fx_model, new_graph)


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def _replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)
    return new_module


def _matches_module_pattern(pattern: Iterable[Type], node: fx.Node, in_node: fx.Node, modules: Dict[str, Any]):
    # if len(node.args) == 0:
    #     return False
    nodes: Tuple[Any, fx.Node] = (in_node, node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


class QuantizationModelWrapper:
    def __init__(self, in_model, in_cc: CompressionConfig, in_econfig: EptqConfig):
        print("Start BN Start")
        model_fold = fuse(in_model)
        print("End BN Fuse")
        print("Starting Layer Wrapping")
        last_linear_layer = [n for n, m in model_fold.named_modules()
                             if isinstance(m, tuple([t[0] for t in LINEAR_OPS]))][-1]
        self.qmodel = replace2quantized_model(model_fold, in_cc,
                                              linear_patterns=LINEAR_OPS,
                                              act_patterns=ACTIVATION_OPS,
                                              is_act_quantization=in_cc.enable_act_quantization,
                                              last_linear_layer=last_linear_layer,
                                              in_econfig=in_econfig)

        self.qmodel.train(False)

        print("End Layer Wrapping")
        self.cc = in_cc
        self.econfig = in_econfig

    def __call__(self, x):
        return self.qmodel(x)

    def apply_weights_quantization(self):
        print("Apply Quantization")
        for n, m in tqdm(self.qmodel.named_modules(), desc='Apply quantization to modules'):
            if isinstance(m, QuantizedWrapper):
                m.apply_weights_quantization()

    def compute_lfh(self, in_images, h_n_iter=100):
        hc = HessianConfig(n_iter=h_n_iter)
        results = weight_lfh(in_images, self.qmodel, hc)

        for n, m in tqdm(self.qmodel.named_modules()):
            if isinstance(m, QuantizedWrapper):
                m.add_hessian_information(results[n])

    def init_module_reconstruction(self):
        self.qmodel.train()
        for n, m in tqdm(self.qmodel.named_modules()):
            if isinstance(m, QuantizedWrapper):
                m.init_module_reconstruction()
            if self.cc.qdrop and isinstance(m, ActivationQuantizedWrapper):
                m.active_qdrop = self.cc.qdrop

    def set_first_last_eight_bit(self, first_output = False, disable_qdrop_first_act = False):
        quant_modules = [m for _, m in self.qmodel.named_modules() if isinstance(m, QuantizedWrapper)]
        quant_modules[0].w_n_bits = 8
        quant_modules[0].include_in_mp = False
        quant_modules[-1].w_n_bits = 8
        quant_modules[-1].include_in_mp = False

        if self.cc.enable_act_quantization:
            last_act = [m for _, m in self.qmodel.named_modules() if isinstance(m, ActivationQuantizedWrapper)][-1]
            last_act.a_n_bits = 8
            if disable_qdrop_first_act:
                first_act = [m for _, m in self.qmodel.named_modules() if isinstance(m, ActivationQuantizedWrapper)][0]
                first_act.disable_qdrop = True
            if first_output:
                first_act = [m for _, m in self.qmodel.named_modules() if isinstance(m, ActivationQuantizedWrapper)][0]
                first_act.a_n_bits = 8

    def set_model_activation_qparams_search(self, v):
        for n, m in self.qmodel.named_modules():
            if isinstance(m, ActivationQuantizedWrapper):
                m.set_layer_qparams_search(v)

    def reset_activation_drop_probability(self):
        for n, m in self.qmodel.named_modules():
            if isinstance(m, ActivationQuantizedWrapper):
                m.active_qdrop = False
