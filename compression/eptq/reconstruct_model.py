from typing import Any

import torch
from torch.nn import Conv2d, Linear
from tqdm import tqdm

from compression import QuantizationModelWrapper
from compression.act_quantized_wrapper import ActivationQuantizedWrapper
from compression.eptq.eptq_config import EptqConfig
from compression.eptq.loss_functions import LFHLossFunctionHessianPerImage
from compression.hessian.per_image_hessian import generate_hessian_scale_dataset
from compression.quantized_wrapper import QuantizedWrapper


def run_eptq(qm: QuantizationModelWrapper, float_model: torch.nn.Module, econfig: EptqConfig,
             num_batches: float, dataloader: Any, device: Any):

    trainable_layers = [module for module in qm.qmodel.modules() if isinstance(module, QuantizedWrapper)]
    act_trainable_layers = [module for module in qm.qmodel.modules() if isinstance(module, ActivationQuantizedWrapper)]

    cali_data = generate_hessian_scale_dataset(qm, dataloader)

    #################################
    # Set Optimizers
    #################################

    ## Weights rounding
    opt_params = []
    for m in trainable_layers:
        opt_params.append(m.alpha)

    optimizer = torch.optim.RAdam(opt_params, lr=econfig.lr)

    ## Bias training
    if econfig.bias_train:
        opt_bias = [module.bias for module in trainable_layers if module.bias is not None]
        optimizer_bias = torch.optim.Adam(opt_bias, lr=econfig.lr_bias)
    else:
        optimizer_bias = None

    ## Weights threshold training
    if econfig.qparam_train:
        opt_qparam = []
        for module in trainable_layers:
            opt_qparam.append(module.get_parameter(f"{module.name.replace('.', '_')}_scale"))
        optimizer_qparam = torch.optim.Adam(opt_qparam, lr=econfig.lr_qparam)
    else:
        optimizer_qparam = None

    ## Activation scale training
    if qm.cc.act_lsq:
        act_scale = [module.scale for _, module in qm.qmodel.named_modules()
                     if isinstance(module, ActivationQuantizedWrapper)]
        act_scale_opt = torch.optim.Adam(act_scale, lr=econfig.act_scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(act_scale_opt, T_max=econfig.n_iters, eta_min=0.)
    else:
        act_scale_opt = None
        a_scheduler = None

    #################################
    # Set Loss function
    #################################

    q_handles = []
    f_handles = []

    loss_func = LFHLossFunctionHessianPerImage(trainable_layers=trainable_layers,
                                               act_trainable_layers=act_trainable_layers,
                                               b_range=(20, 2), decay_start=0, warmup=0.2, cc=qm.cc,
                                               econfig=econfig)

    # register hook for knowledge distillation loss on float and quant models
    model_register_hook(qm.qmodel, loss_func.q_activation_tensors, handles=q_handles, layer_types=QuantizedWrapper)
    model_register_hook(float_model, loss_func.f_activation_tensors, handles=f_handles, layer_types=(Conv2d, Linear))


    #################################
    # Run optimization loop
    #################################
    n_epochs = econfig.n_iters // num_batches
    for _ in tqdm(range(n_epochs), "EPTQ Optimization..."):
        for data, hessian in cali_data:
            cur_inp = data.to(device)
            hessian = {k: v.to(device) for k, v in hessian.items()}

            optimizer.zero_grad()

            if qm.cc.act_lsq:
                act_scale_opt.zero_grad()

            if econfig.bias_train:
                optimizer_bias.zero_grad()

            if econfig.qparam_train:
                optimizer_qparam.zero_grad()

            _ = qm.qmodel(cur_inp)
            _ = float_model(cur_inp)

            err, rec, roundloss, reg_roundloss, b = loss_func(hessian)

            err.backward(retain_graph=True)

            optimizer.step()

            if qm.cc.act_lsq:
                act_scale_opt.step()
                a_scheduler.step()

            if econfig.bias_train:
                optimizer_bias.step()

            if econfig.qparam_train:
                optimizer_qparam.step()

            qm.econfig.increment_eptq_iter()

    # remove KD handles (if exist):
    for h in q_handles:
        h.remove()
    for h in f_handles:
        h.remove()


def model_register_hook(in_net, layer2tensor, handles, layer_types):
    def get_activation(in_name):
        def hook(model, input, output):
            layer2tensor.append((in_name, output.clone()))

        return hook

    for name, module in in_net.named_modules():
        if isinstance(module, layer_types):
            if '.act_op' in name:
                # If activations are quantized, then we want to take the activation tensor after the conv op before
                # the quantization, so we need to edit the layer name for convenience in the comparison later
                name = name.split(".act_op")[0]
            handles.append(module.register_forward_hook(get_activation(name)))
