import torch

def OD_collate(batch):
    images, targets, idx = tuple(zip(*batch))
    return(tuple((torch.stack(images).float(), targets, idx)))
    #return (tuple(zip(*batch))

def collate(name):
    f = globals().get(name)
    return(f)
