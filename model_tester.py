import torch
import os
from tqdm import tqdm
import numpy as np
import argparse

from model.gen_model import get_test

def parse_args():
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('--model', dest='model_zoo',
                        help='model zoo name',
                        default=None, type=str)
    return parser.parse_args()


def test_model(model):
    """
    Input: Model
    Function: Test Model with fake inputs
    """
    image = torch.randn(1, 3, 800, 800)
    target = [{'boxes': torch.tensor([[0, 0, 1, 1]]), 'labels':
    torch.tensor([0])}]
    print("Input shape (C, H, W), {}".format(image.shape))
    print('#'*50)
    print('Model in training mode')
    print(model(image, target))
    print('#'*50)
    print('Model in eval mode')
    model.eval()
    print(model(image))


if __name__ == '__main__':
    args = parse_args()
    model_name = str(args.model_zoo)
    model = get_test(model_name)
    test_model(model)

