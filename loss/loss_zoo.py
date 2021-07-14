import numpy as np

def placeholder(boxes, scores, config):
    return()

def loss(name):
    f = globals().get(name)
    return f


if __name__ == '__main__':
    print(globals())
