import torch
import os
from tqdm import tqdm
import numpy as np
import argparse
import utils
import utils.config
import utils.device
import utils.input 
import utils.checkpoint
from utils.logs import Logman
from model.gen_model import get_model
from dataset.gen_dataloader import get_dataloader
from optimiser.gen_optimiser import get_optimiser
from scheduler.gen_scheduler  import get_scheduler
from transform.gen_transform import get_transform
import math

def train_one_cycle(config, model, dataloader, optimiser, epoch, device):
    """
    Run one epoch of training, backpropogation and optimisation.
    """
    # model train mode
    model.train()

    batch_size = config['train']['batch_size']
    len_dataset = len(dataloader.dataset)
    step = math.ceil(len_dataset / batch_size)
    # progress bar
    train_prog_bar = tqdm(dataloader, total=step)
    running_loss = 0

    with torch.set_grad_enabled(True):
        for batch_num, (images, targets, idx) in enumerate(train_prog_bar):
            # zero gradient optim
            optimiser.zero_grad()
            # send to devices
            images = images.to(device)
            tg = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # get outputs
            losses = model(images, tg)
            # training
            train_loss = sum(loss for loss in losses.values())

            # Backpropogation
            train_loss.backward()
            optimiser.step()
            # For averaging and reporting later
            running_loss += train_loss.item()
            # logging
            if logman:
                logman.log({'type': 'train', 
                            'epoch': epoch, 
                            'batch': batch_num, 
                            'loss': train_loss.item()
                        })
            # show the current loss to the progress bar
            train_pbar_desc = f'loss: {train_loss.item():.4f}'
            train_prog_bar.set_description(desc=train_pbar_desc)

        # average the running loss over all batches and return
        train_running_loss = running_loss / step
        print(f"Final Training Loss: {train_running_loss:.4f}")

        # free memory
        del images, tg, losses, train_loss
        # free up cache
        torch.cuda.empty_cache()
        return(train_running_loss)

def val_one_cycle(config, model, dataloader, optimiser, epoch, device):
    """
        Runs one epoch of prediction.
        In model.train() mode, model(images)  is returning losses.
        We are using model.eval() mode --> it will return boxes and scores.
     """
    model.eval()
    batch_size = config['train']['batch_size']
    len_dataset = len(dataloader.dataset)
    step = math.ceil(len_dataset / batch_size)
    valid_prog_bar = tqdm(dataloader, total=step)
    running_prec = 0
    with torch.no_grad():

        metric = 0

        for batch_num, (images, targets, idx) in enumerate(valid_prog_bar):
            # send to devices
            images = images.to(device)
            # get predictions
            outputs = model(images)
            # get metric
            for i, image in enumerate(images):
                gt_boxes = targets[i]['boxes'].data.cpu().numpy()
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].detach().cpu().numpy()
                precision=1
                avg=1
                # logging
                if logman:
                    logman.log({'type': 'val', 
                                'epoch': epoch, 
                                'batch': batch_num, 
                                'image_precision': precision,
                                'average_precision': avg
                            })
                # Show the current metric
                valid_pbar_desc = f"Current Precision: {precision:.4f}"
                valid_prog_bar.set_description(desc=valid_pbar_desc)
                running_prec += precision
        final_prec = running_prec / step      
        print(f"Validation metric: {final_prec:.4f}")
        # Free up memory
        del images, outputs, gt_boxes, boxes, scores ,precision
        torch.cuda.empty_cache()
        return(final_prec)


def train(config, model, dataloaders, optimiser, scheduler, device):
    num_epochs = config['train']['num_epochs']
    for epoch in range(num_epochs):
        # train
        final_loss = train_one_cycle(config, model, dataloaders['train'],
                        optimiser, epoch, device)
        # val
        final_prec = val_one_cycle(config, model, dataloaders['val'],
                      optimiser, epoch, device)
        # scheduler
        if scheduler:
            scheduler.step()
        utils.checkpoint.save(config, model, optimiser, scheduler, epoch, final_loss, final_prec)
    # end logging
    logman.log({'type': 'final'})

def run(config):
    # directories
    input_dir = config['data']['input']
    DEVICE = utils.device.get_device()
    # get elements
    model = get_model(config).to(DEVICE)
    optimiser = get_optimiser(config, model.parameters())
    scheduler = get_scheduler(config, optimiser, -1)
    df = utils.input.get_dfs(config)
    dataloaders = {split:get_dataloader(config, df, split, get_transform(config, split))
                   for split in ['train', 'val']}
    train(config, model, dataloaders, optimiser, scheduler, DEVICE)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = utils.config.load(args.config_file)
    print(config)
    logman = Logman(config, config['output']['dir'], config['model']['name'])
    run(config)