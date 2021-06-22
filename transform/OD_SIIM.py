# Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

def train_t():
    return(A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.4,
                                  contrast_limit=0.4, p=0.3),
                 A.RandomGamma(gamma_limit=(50, 300), eps=None, always_apply=False,
                      p=0.3)], p=0.2),
        A.Resize(height=Config.reshape_size[0], width=Config.reshape_size[1],
                 p=1.0),
        ToTensorV2(p=1.0)],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}))




def val_t():
    return(A.Compose([A.Resize(height=Config.reshape_size[0],
                                      width=Config.reshape_size[1], p=1.0),
                             ToTensorV2(p=1.0)],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}))
def get_transform(split):
    if split == 'train':
        return(train())
    elif spit == 'val':
        return(val_t():
