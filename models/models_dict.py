import torch

import helpers
from models.resnet import resnet18 as resnet18_qdrop
from models.resnet import resnet50 as resnet50_qdrop
from models.mobilenetv2 import mobilenetv2 as mobilenetv2_qdrop
from models.regnet import regnetx_600m as regnetx_600m_qdrop
from models.regnet import regnetx_3200m as regnetx_3200m_qdrop
from models.mnasnet import mnasnet as mnasnet_qdrop


MODELS_DICT = {
    'resnet18': (resnet18_qdrop, 71.002),
    'resnet50': (resnet50_qdrop, 76.626),
    'mobilenetv2': (mobilenetv2_qdrop, 72.622),
    'regnet_600m': (regnetx_600m_qdrop, 73.514),
    'regnet_3200m': (regnetx_3200m_qdrop, 78.468),
    'mnasnet': (mnasnet_qdrop, 76.528),
}


def load_qdrop_model(model, checkpoints, model_name=None, **kwargs):
    model = model(**kwargs)
    state_dict = torch.load(checkpoints, map_location=helpers.get_device())
    if model_name == 'mobilenetv2':
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    return model


def load_model(model_name, checkpoints):
    model, float_acc = MODELS_DICT[model_name]
    kwargs = {'num_classes': 1000} if model_name in ['resnet18', 'resnet50', 'mobilenetv2'] else {}
    loaded_model = load_qdrop_model(model, checkpoints, model_name=model_name, **kwargs)

    return loaded_model, float_acc
