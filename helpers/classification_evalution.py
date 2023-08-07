import torch
from helpers.device_handler import to_device
from tqdm import tqdm


def classification_evaluation(in_model, in_dataloader):
    count = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(in_dataloader):
            x, y = to_device(x, y)
            y_hat = in_model(x)
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            count += x.shape[0]
    acc = (correct / count) * 100
    print(f"Accuracy:{acc}")
    return acc
