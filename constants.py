import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LINEAR_OPS = [(torch.nn.Conv1d,),
              (torch.nn.Conv2d,),
              (torch.nn.Conv3d,),
              (torch.nn.Linear,)]

ACTIVATION_OPS = [(torch.nn.ReLU,),
                  (torch.nn.ReLU6,),
                  (torch.nn.Identity,)]


SIGMOID_MINUS = 4
