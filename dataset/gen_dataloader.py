from torch.utils.data import DataLoader
import os
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


def get_train(train_test):
    """
    get train params from config
    """
    train_ids = train_test['train']['image_ids']
    train_transform = train_test['train']['transform']
    return(train_ids, train_transform)


def get_val(train_test):
    """
    get train params from config
    """
    val_ids = train_test['valid']['image_ids']
    val_transform = train_test['valid']['transform']
    return(val_ids, val_transform)


def get_train_val(config, df, train_test):
    """
    Input config, the right df and a dictionary of train_test.

    train_test = {valid: {'image_ids': list, 'transform': val_transforms'},
                  train: {'image_ids': list, 'transform': train_transforms'},
                 }

    """
    if config.collate_name:
        collate_fn = get_collate(config['collate_name'])
    else:
        collate_fn = None
    # train
    train_ids, train_tranform = get_train(train_test)
    train_dataset = get_dataset(config['dataset']['name'], df, train_ids, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config.T_BS,
                                  shuffle=True, collate_fn=collate_fn)
    # val
    val_ids, val_tranform = get_val(train_test)
    val_dataset = get_dataset(config['dataset'], df, val_ids, val_transform)
    val_dataloader = DataLoader(train_dataset, batch_size=config.V_BS,
                                  shuffle=False, collate_fn=collate_fn)

    return(train_dataloader, val_dataloader)
