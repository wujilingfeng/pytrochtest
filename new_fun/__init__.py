import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import init
#import my_fun
print("new_fun is imported")
#import load
from .load import get_batch,unpickle,show_image
__all__=["np","torch","nn","Parameter","Function","F","unpickle","get_batch","my_fun","show_image","init"]

#现在的new_fun包，包含np,torch,nn,Parameter,Function F,my_fun,load
