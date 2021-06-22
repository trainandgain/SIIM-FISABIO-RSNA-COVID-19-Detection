from train_transform import train_transfrom
from val_transform import val_transform

def get_transform(config, split, params=None):
  f = globals().get(config['transform']['name'])
  if params is not None:
    return f(split, **params)
  else:
    return f(split, **config['transform']['name'])
