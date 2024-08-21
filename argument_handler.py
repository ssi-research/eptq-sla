import argparse

from compression import ThresholdMethod
from compression.eptq.eptq_config import AnnealingFunction
from models.models_dict import MODELS_DICT


def argument_handler():
    #################################
    ######### Run Arguments #########
    #################################

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help='The name of the model to run', choices=MODELS_DICT.keys())
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--model_checkpoints', type=str, required=True)
    parser.add_argument('--project_name', type=str, default='eptq')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--eval_float_accuracy', action='store_true')
    parser.add_argument('--comment', nargs='+', type=str, default=None)

    #################################
    #### Quantization Parameters ####
    #################################

    parser.add_argument('--weights_nbits', type=int, default=8)
    parser.add_argument('--activation_nbits', type=int, default=8)
    parser.add_argument('--disable_activation_quantization', action='store_true', default=False)
    parser.add_argument('--threshold_method', type=str, default='HMSE',
                        choices=[i.name for i in ThresholdMethod])
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--act_calib_num_samples', type=int, default=256)
    parser.add_argument('--mse_p', type=float, default=2.0)
    parser.add_argument('--disable_first_last_eight_bit', action='store_true', default=False)

    ####################################
    ####### Hessians Parameters ########
    ####################################
    parser.add_argument('--h_w_num_samples', type=int, default=64)
    parser.add_argument('--h_n_iter', type=int, default=100)

    #########################
    #### EPTQ Parameters ####
    #########################
    parser.add_argument('--disable_eptq', action='store_true', default=False)
    parser.add_argument('--disable_bias_train', action='store_true', default=False)
    parser.add_argument('--disable_qparam_train', action='store_true', default=False)
    parser.add_argument('--disable_act_lsq', action='store_true', default=False)
    parser.add_argument('--act_scale_lr', type=float, default=4e-5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_bias', type=float, default=1e-4)
    parser.add_argument('--lr_qparam', type=float, default=1e-4)
    parser.add_argument('--reg_factor', type=float, default=1e-2)
    parser.add_argument('--eptq_iters', type=int, default=80000)
    parser.add_argument('--b_decay_mode', type=str, default='linear')

    ## QDROP
    parser.add_argument('--disable_qdrop', action='store_true', default=False)
    parser.add_argument('--drop_start', type=float, default=1.0)
    parser.add_argument('--drop_end', type=float, default=0.0)
    parser.add_argument('--vanilla_qdrop', action='store_true', default=False)
    parser.add_argument('--disable_grad_act_quant', action='store_true', default=False)
    parser.add_argument('--prob_warmup', type=float, default=0.0)
    parser.add_argument('--prob_start_b', type=int, default=0)
    parser.add_argument('--prob_end_b', type=int, default=20)
    parser.add_argument('--annealing_fn', type=str, default='LINEAR',
                        choices=[i.name for i in AnnealingFunction])

    parser.add_argument('--disable_first_act', action='store_true', default=False)
    parser.add_argument('--disable_qdrop_first_act', action='store_true', default=False)

    args = parser.parse_args()
    return args
