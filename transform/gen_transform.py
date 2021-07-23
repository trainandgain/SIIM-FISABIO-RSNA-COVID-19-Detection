# Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

def train(config):
    return(A.Compose([A.RandomRotate90(p=0.1), A.Flip(p=0.1),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.Resize(height=800, width=800, p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])))


def val(config):
    return(A.Compose([A.Resize(height=800, width=800, p=1.0)], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])))



def get_transform(config, split, params=None):
  f = globals().get(split)
  if params is not None:
    return f(config, **params)
  else:
    return f(config)
