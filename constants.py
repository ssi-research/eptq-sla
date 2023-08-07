import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Local copies:
if os.path.isdir('/local_datasets/ImageNet/ILSVRC2012_img_val_TFrecords'):
    VAL_DIR = '/local_datasets/ImageNet/ILSVRC2012_img_val_TFrecords'
    TRAIN_DIR = '/local_datasets/ImageNet/ILSVRC2012_img_train'
else:
    # Netapp copies:
    VAL_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords'
    TRAIN_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train'
