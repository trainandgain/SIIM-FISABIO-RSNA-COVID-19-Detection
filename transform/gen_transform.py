# Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

def OD(config, split):
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
          A.Resize(height=config['transform']['height'], width=config['transform']['width'], p=1.0)],
          bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])))


  def val(config):
      return(A.Compose([A.Resize(height=config['transform']['height'], width=config['transform']['width'], p=1.0)], 
      bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])))
  
  if split=='train':
    return(train(config))
  else:
    return(val(config))

def IC(config, split):
  def train(config):
      return(A.Compose([
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.4,
                                  contrast_limit=0.4, p=0.3),
                 A.RandomGamma(gamma_limit=(50, 300), eps=None, always_apply=False, 
                      p=0.3),
            ], p=0.2),
        A.Resize(height=config['transform']['height'], width=config['transform']['width'], 
                 p=1.0)]))


  def val(config):
      return(A.Compose([A.Resize(height=config['transform']['height'], 
      width=config['transform']['width'], p=1.0)]))
  
  if split=='train':
    return(train(config))
  else:
    return(val(config))


def get_transform(config, split):
  f = globals().get(config['transform']['name'])
  return(f(config, split))
