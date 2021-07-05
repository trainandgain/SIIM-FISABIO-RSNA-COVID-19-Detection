import torch
from torch import nn
import torchvision
import math
from torch import nn, Tensor
from typing import List, Tuple, Dict, Optional


def model(name):
    f = globals().get(name)
    return f()


if __name__ == '__main__':
    print(globals())
