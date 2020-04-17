import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()

def variable_fun(device):
    return lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(device) \
        if USE_CUDA else autograd.Variable(*args, **kwargs)

