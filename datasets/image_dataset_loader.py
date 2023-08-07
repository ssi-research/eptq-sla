import random

from torch.utils.data import ConcatDataset
from torchvision import datasets
import torch
from torchvision.transforms import transforms

from constants import TRAIN_DIR
from constants import VAL_DIR


def random_crop_flip_preprocess():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def validation_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])





def get_imagenet_dataset_loader(batch_size):
    preprocess = validation_preprocess()
    val_dataset = datasets.ImageFolder(VAL_DIR, preprocess)

    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
