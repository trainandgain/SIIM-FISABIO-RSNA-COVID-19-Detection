import warnings
import argparse
import utils
import utils.config
import utils.device
import utils.input 
import utils.checkpoint
from utils.logs import Logman
from model.gen_model import get_model
from metric.gen_metric import get_metric
from loss.gen_loss import get_loss
from dataset.gen_dataloader import get_dataloader
from optimiser.gen_optimiser import get_optimiser
from scheduler.gen_scheduler  import get_scheduler
from transform.gen_transform import get_transform
from loop.gen_loop import get_loop
import math

def run(config):
    DEVICE = utils.device.get_device()
    # get elements
    model = get_model(config).to(DEVICE)
    optimiser = get_optimiser(config, model.parameters())
    scheduler = get_scheduler(config, optimiser, -1)
    metric = get_metric(config)
    loop = get_loop(config)
    if config['loss']:
        loss = get_loss(config)
    df = utils.input.get_dfs(config)
    dataloaders = {split:get_dataloader(config, df, split, get_transform(config, split))
                   for split in ['train', 'val']}
    loop(config, model, dataloaders, optimiser, scheduler, DEVICE, metric, logman, loss)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    config = utils.config.load(args.config_file)
    print(config)
    logman = Logman(config, config['output']['dir'], config['model']['name'])
    run(config)