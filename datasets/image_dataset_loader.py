from torch.utils.data import ConcatDataset
from torchvision import datasets
import torch
from torchvision.transforms import transforms


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


def get_init_rep_dset(args):
    init_dataset = datasets.ImageFolder(args.train_dir, random_crop_flip_preprocess())

    return init_dataset


def get_representative_dataset(args, shuffle):
    init_dataset = get_init_rep_dset(args)
    representative_data_loader = torch.utils.data.DataLoader(init_dataset,
                                                             batch_size=args.batch_size,
                                                             shuffle=shuffle,
                                                             num_workers=4,
                                                             pin_memory=True)
    return representative_data_loader


def get_train_samples(train_loader, num_samples):
    train_data = []
    labels = []
    for batch in train_loader:
        train_data.append(batch[0])
        labels.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    random_samples_indices = torch.randperm(num_samples)[:num_samples]
    return torch.cat(train_data, dim=0)[random_samples_indices], torch.cat(labels, dim=0)[random_samples_indices]
