import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F
import random


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def softmax_with_temperature(logits, temperature):
    # assert temperature > 0, "temperature必须大于0"
    return F.softmax(logits / temperature, dim=1)


def init(module):
    for child_module in module.children():
        # if isinstance(child_module, nn.Linear):
            # nn.init.xavier_normal_(child.weight)
            # nn.init.xavier_uniform_(child_module.weight)
        for child in child_module.children():
            if isinstance(child, nn.Linear):
                # nn.init.xavier_normal_(child.weight)
                nn.init.xavier_uniform_(child.weight)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False