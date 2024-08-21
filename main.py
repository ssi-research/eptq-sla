import copy

import torch
from torch.fx.experimental.optimization import fuse
from torch.utils.data import Subset
import numpy as np
import timm

import helpers
from argument_handler import argument_handler
from compression import QuantizationModelWrapper, CompressionConfig, ThresholdMethod
from compression.eptq.eptq_config import EptqConfig, QdropConfig, AnnealingFunction
from compression.eptq.reconstruct_model import run_eptq
from datasets.image_dataset_loader import get_train_samples
from helpers.set_seed import set_seed
from models.models_dict import load_model

if __name__ == '__main__':
    args = argument_handler()
    set_seed(args.random_seed)

    # Load Dataset
    transform = timm.data.create_transform(224, interpolation="bilinear", color_jitter=None, is_training=True)
    ds = timm.data.create_dataset("", args.train_dir, transform=transform)
    sub_ds = Subset(ds, list(np.random.randint(0, len(ds) + 1, args.num_samples)))

    train_dataloader = torch.utils.data.DataLoader(sub_ds, args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = timm.data.create_loader(timm.data.create_dataset("", args.val_dir), 224, args.batch_size,
                                             use_prefetcher=False, interpolation="bilinear")

    print(f"Generating {args.num_samples} calibration samples")
    calib_samples, calib_labels = get_train_samples(train_dataloader, num_samples=args.num_samples)

    # Load Model
    model, float_acc = load_model(args.model_name, args.model_checkpoints)
    model = model.to(helpers.get_device())
    model.eval()

    if args.eval_float_accuracy:
        desc = "Evaluating float accuracy"
        float_acc = helpers.classification_evaluation(model, val_dataloader, desc=desc)

    threshold_method = ThresholdMethod[args.threshold_method]

    # Set Configs
    cc = CompressionConfig(w_n_bits=args.weights_nbits, threshold_method=threshold_method,
                           mse_p=args.mse_p, qparam_train=not args.disable_qparam_train,
                           first_last_eight_bit=not args.disable_first_last_eight_bit, a_n_bits=args.activation_nbits,
                           enable_act_quantization=not args.disable_activation_quantization,
                           act_lsq=not args.disable_act_lsq, qdrop=not args.disable_qdrop)

    qdrop_config = None if args.disable_qdrop else QdropConfig(drop_start=args.drop_start, drop_end=args.drop_end,
                                                               grad_act_quant=not args.disable_grad_act_quant,
                                                               vanilla_qdrop=args.vanilla_qdrop,
                                                               annealing_fn=AnnealingFunction[args.annealing_fn])

    econfig = None if args.disable_eptq else EptqConfig(lr=args.lr, n_iters=args.eptq_iters,
                                                        bias_train=not args.disable_bias_train,
                                                        lr_bias=args.lr_bias,
                                                        qparam_train=not args.disable_qparam_train,
                                                        lr_qparam=args.lr_qparam,
                                                        num_samples=args.num_samples,
                                                        reg_factor=args.reg_factor, act_scale_lr=args.act_scale_lr,
                                                        qdrop_config=qdrop_config, b_decay_mode=args.b_decay_mode)

    # Quantize Model
    qm = QuantizationModelWrapper(model, cc, in_econfig=econfig)

    if not args.disable_first_last_eight_bit:
        qm.set_first_last_eight_bit(first_output=not args.disable_first_act,
                                    disable_qdrop_first_act=args.disable_qdrop_first_act)

    if threshold_method == ThresholdMethod.HMSE:
        in_images = calib_samples[:args.h_w_num_samples]
        qm.compute_lfh(in_images, h_n_iter=args.h_n_iter)

    qm.apply_weights_quantization()

    if cc.enable_act_quantization:
        act_calib_samples = calib_samples[:args.act_calib_num_samples].to(helpers.get_device())
        # run on batches
        print(f"Initializing Activation quantization parameters with {args.act_calib_num_samples} samples")
        qm.set_model_activation_qparams_search(True)
        with torch.no_grad():
            qm.qmodel(act_calib_samples)
        qm.set_model_activation_qparams_search(False)

    # EPTQ
    if not args.disable_eptq:
        cc.set_weights_quantization(quantize_weights=True)
        cc.set_act_quantization(quantize_activation=cc.enable_act_quantization)
        qm.init_module_reconstruction()

        print(f"Running EPTQ optimization with {econfig.num_samples} samples")
        float_model = copy.deepcopy(model).to(helpers.get_device())
        float_model = float_model.eval()
        float_model = fuse(float_model)

        num_batches = econfig.num_samples // args.batch_size

        run_eptq(qm=qm, float_model=float_model, econfig=econfig, num_batches=num_batches, device=helpers.get_device(),
                 dataloader=train_dataloader)

        if cc.qdrop:
            qm.reset_activation_drop_probability()

    else:
        cc.set_weights_quantization(quantize_weights=True)
        cc.set_act_quantization(quantize_activation=cc.enable_act_quantization)

    qm.qmodel.eval()

    desc = "Evaluating quant accuracy"
    quant_acc = helpers.classification_evaluation(qm, val_dataloader, desc=desc)

    logs = {'float_accuracy': float_acc,
            'quant_accuracy': quant_acc}

    print(logs)
