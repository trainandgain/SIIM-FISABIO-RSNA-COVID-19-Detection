import torch
import os
def save(config, model, optim, scheduler, epoch):
    save_path = config['output']['dir'] +'/'+config['model']['name']+'/'
    # save model
    model.eval()
    torch.save({
        'model_state_dict': model.state_dict(), #'model_state_dict': self.model.model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': current_train_loss,
        'image_precision': current_precision,
        'epoch': epoch,
        
    }, save_path+'epoch_{epoch}.pth'.format(epoch=epoch))

