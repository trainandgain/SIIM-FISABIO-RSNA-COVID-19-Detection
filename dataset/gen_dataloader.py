from torch.utils.data import DataLoader
import os
import pandas as pd
import gen_dataset as gen_dataset
import gen_collate as gen_collate

def get_dataset(name, df, image_ids, transform=None):
   """
    Using the config file we grab the correct dataset.
    Then create a valid and train dataloader.
   """
   return(gen_dataset.dataset(name, image_ids, df, transform)


def get_collate(name):
    return(gen_collate.collate(name))

def get_dataloader(config, df, fold, transform=None):
    """
    Input config and split.
    """
    if config.collate_name:
        collate_fn = get_collate(config['collate_name'])
    else:
        collate_fn = None
    if split == 'train':
        id_s = df[df.fold!=config['dataset']['val_fold']
    else:
        id_s = df[df.fold==config['dataset']['val_fold']
    dataloader = get_dataset(config['dataset']['name'], df, id_s, transform)
    return(dataloader)
