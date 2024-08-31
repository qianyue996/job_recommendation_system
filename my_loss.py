import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(outputs, label1, label2):
        loss1 = nn.CrossEntropyLoss()(outputs[0], label1)
        loss2 = nn.CrossEntropyLoss()(outputs[1], label2)
        return 5*loss1 + loss2