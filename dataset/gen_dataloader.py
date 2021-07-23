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

def get_dataloader(config, df, split, transform=None):
    """
    Input config and split.
    """
    if config['data']['collate_name'] != 'None':
        collate_fn = get_collate(config['data']['collate_name'])
    else:
        collate_fn = None
    if split == 'train':
        id_s = df[df.fold!=config['data']['params']['val_fold']].id.values[:10]
    else:
        id_s = df[df.fold==config['data']['params']['val_fold']].id.values[:10]
    dataloader = DataLoader(get_dataset(config['data']['name'], df, id_s, transform),
                            collate_fn=collate_fn, 
                            batch_size=config['train']['batch_size'])                    
    return(dataloader)
