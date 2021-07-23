import torch
import math
from tqdm import tqdm
import utils
import tracemalloc
import gc

def OD(config, model, dataloaders, optimiser, scheduler, device, metric, logman, loss=None):
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
                # gradient clipping
                if config['train']['gradient_clipping']:
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                    **config['train']['gradient_clipping']['params'])
                # optimiser step
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
            images, tg, losses, train_loss = None, None, None, None
            del images, tg, losses, train_loss
            # free up cache
            torch.cuda.empty_cache()
            gc.collect()
            return(train_running_loss)

    def val_one_cycle(config, model, dataloader, optimiser, epoch, device, metric):
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
                    precision = metric(boxes, scores, gt_boxes, config)
                    running_prec += precision
                    # logging
                    if logman:
                        logman.log({'type': 'val', 
                                    'epoch': epoch, 
                                    'batch': batch_num, 
                                    'image_precision': precision,
                                })
                    # Show the current metric
                    valid_pbar_desc = f"Current Precision: {precision:.4f}"
                    valid_prog_bar.set_description(desc=valid_pbar_desc)
                    running_prec += precision
                    
            final_prec = running_prec / step      
            print(f"Validation metric: {final_prec:.4f}")
            # Free up memory
            images, outputs, gt_boxes, boxes, scores, precision = None, None, None, None, None, None
            del images, outputs, gt_boxes, boxes, scores, precision
            torch.cuda.empty_cache()
            gc.collect()
            return(final_prec)


    def train(config, model, dataloaders, optimiser, scheduler, device, metric):
        num_epochs = config['train']['num_epochs']
        for epoch in range(num_epochs):
            # train
            final_loss = train_one_cycle(config, model, dataloaders['train'],
                            optimiser, epoch, device)
            # val
            final_prec = val_one_cycle(config, model, dataloaders['val'],
                        optimiser, epoch, device, metric)
            # scheduler
            if scheduler:
                scheduler.step()
            utils.checkpoint.save(config, model, optimiser, scheduler, epoch, final_loss, final_prec)
        # end logging
        logman.log({'type': 'final'})

    train(config, model, dataloaders, optimiser, scheduler, device, metric)


def IC(config, model, dataloaders, optimiser, scheduler, device, metric, logman, loss=None):
    def train_one_cycle(config, model, dataloader, optimiser, epoch, device, loss):
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
            for batch_num, (images, labels, idx) in enumerate(train_prog_bar):
                # zero gradient optim
                optimiser.zero_grad()
                # send to devices
                images = images.to(device)
                labels = labels.to(device)
                # get outputs
                out = model(images)
                # training
                train_loss = loss(out, labels.float())
                # Backpropogation
                train_loss.backward()
                # gradient clipping
                if config['train']['gradient_clipping']:
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                    **config['train']['gradient_clipping']['params'])
                # optimiser step
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
            del images, labels, out
            # free up cache
            torch.cuda.empty_cache()
            gc.collect()
            return(train_running_loss)

    def val_one_cycle(config, model, dataloader, optimiser, epoch, device, loss, metric):
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
        running_loss = 0
        all_valid_labels = []
        all_valid_preds = []
        with torch.no_grad():
            for batch_num, (images, labels, idx) in enumerate(valid_prog_bar):
                # send to devices
                images = images.to(device)
                labels = labels.to(device)
                # get predictions
                out = model(images)
                # loss fn
                val_loss = loss(out, labels.float())
                running_loss += val_loss.item()
                # metric
                all_valid_labels += [out.flatten().detach().cpu().numpy()]
                all_valid_preds += [labels.flatten().detach().cpu().numpy()]
                if logman:
                    logman.log({'type': 'val', 
                                'epoch': epoch, 
                                'batch': batch_num, 
                                'loss': val_loss.item()
                            })
            final_loss = running_loss / step      
            print(f"Validation metric: {final_loss:.4f}")
            accuracy, recall, precision, f1_score = metric(all_valid_labels, all_valid_preds)
            if logman:
                logman.log({'type': 'val_metric', 
                                 'epoch': epoch,
                                 'batch': batch_num, 
                                 'accuracy': accuracy,
                                 'recall': recall,
                                 'precision': precision,
                                 'f1_score': f1_score})
            # Free up memory
            del images, out, accuracy, recall, precision, f1_score, labels
            torch.cuda.empty_cache()
            gc.collect()
            return(final_loss)


    def train(config, model, dataloaders, optimiser, scheduler, device, loss, metric):
        num_epochs = config['train']['num_epochs']
        for epoch in range(num_epochs):
            # train
            final_loss = train_one_cycle(config, model, dataloaders['train'],
                            optimiser, epoch, device, loss)
            current, peak = tracemalloc.get_traced_memory()
            # val
            final_prec = val_one_cycle(config, model, dataloaders['val'],
                        optimiser, epoch, device, loss, metric)
            # scheduler
            if scheduler:
                scheduler.step()
            utils.checkpoint.save(config, model, optimiser, scheduler, epoch, final_loss, final_prec)
        # end logging
        logman.log({'type': 'final'})

    train(config, model, dataloaders, optimiser, scheduler, device, loss, metric)


def loop(name):
    f = globals().get(name)
    return f