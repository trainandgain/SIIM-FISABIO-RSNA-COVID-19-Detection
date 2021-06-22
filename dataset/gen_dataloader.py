from torch.utils.data import DataLoader
import os
import pandas as pd
from .gen_dataset import dataset
from .gen_collate import collate

def get_dataset(name, df, image_ids, transform=None):
   """
    Using the config file we grab the correct dataset.
    Then create a valid and train dataloader.
   """
   return(dataset(name, df, image_ids, transform))


def get_collate(name):
    return(collate(name))

def get_dataloader(config, df, fold, transform=None):
    """
    Input config and split.
    """
    if config.collate_name:
        collate_fn = get_collate(config['collate_name'])
    else:
        collate_fn = None
    if split == 'train':
        id_s = df[df.fold!=config['dataset']['val_fold']]
    else:
        id_s = df[df.fold==config['dataset']['val_fold']]
    dataloader = DataLoader(get_dataset(config['dataset']['name'], df, id_s, transform))
    return(dataloader)
