# Enhanced Post-Training Quantization (EPTQ)

EPTQ is a PyTorch post-training quantization method for CV networks.
It uses a Hessian-guided knowledge distillation loss to optimize the rounding error of the quantized parameters.

<p align="center">
  <img src="images/eptq-sla.svg" width="600">
</p>

## Models

We support a large set of pre-trained models. 
The models are based on the implementation provided by BRECQ [1].

The names of the available models can be found under [models/models_dict.py](./models/models_dict.py).

In order to run a model, you'll need to download the model's checkpoints, as explained in [BRECQ's repository](https://github.com/yhhhli/BRECQ/blob/main/README.md) 
and in the following example:

`wget https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar`

The available models include:

| Model        | Model usage name |
|--------------|-----------------|
| ResNet18     | resnet18        |
| ResNet50     | resnet50        |
| MobileNetV2  | mobilenetv2     |
| RegNet-600M  | regnet_600m     |
| RegNet-3.2GF | regnet_3200m    |
| MnasNet-2.0  | mnasnet         |


## Setup

`pip install -r requirements.txt`

## Usage

`python main.py -m resnet18 --train_dir /path/to/trainind/sampls/dir --val_dir /path/to/validation/sampls/folder 
--model_checkpoints /path/to/model/checkpoints.tar`

This example would execute EPTQ for ResNet18 quantization with 8-bit weights and activation,
using the default hyperparameters detailed in the EPTQ paper.

For faster execution you can reduce the number of optimization steps, using the flag 
`--eptq_iters` (80K by default). 
This might result in a small reduction in the final accuracy results.

[1] Li, Yuhang, et al. "BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction." ICLR 2021.
