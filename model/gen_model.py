import torch
from torch import nn
import torchvision
import model_zoo

def get_model(config):
    model_name = config.model.name
    print('Model Name: ', model_name)
    return(model_zoo.model(name))


def get_test(name='FasterRCNNDetector'):
    print('Model Name: ', name)
    return(model_zoo.model(name))

if __name__ == '__main__':
    model = get_test()
    print(dir(model))

