# Faster RCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection import  FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
# EFFNET
from efficientnet_pytorch import EfficientNet
#normal imports 
import torch
import torchvision
import math
from torch import nn, Tensor
from typing import List, Tuple, Dict, Optional
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import paste_masks_in_image

class FasterRCNNDetector(torch.nn.Module):
    def __init__(self, NUM_CLASSES, pretrained=True, **kwargs):
        super(FasterRCNNDetector, self).__init__()
        # load pre-trained model incl. head
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, 
                                                                          pretrained_backbone=pretrained)
        # get number of input features for the classifier custom head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    def forward(self, images, targets=None):
        return self.model(images, targets)

class EfficientNETB2(nn.Module):
    def __init__(self, NUM_CLASSES, pretrained=True):
        super(EfficientNETB2, self).__init__()
        # model
        self.NUM_CLASSES = NUM_CLASSES
        self.effnet = self.load_effnet(pretrained)
        self.effnet._fc = nn.Linear(1408, self.NUM_CLASSES)
        self.out = nn.Sigmoid()
        
    def load_effnet(self, pretrained):
        if pretrained == True:
            effnet = EfficientNet.from_pretrained("efficientnet-b2")
        else:
            effnet = EfficientNet.from_name("efficientnet-b2")
        return effnet
        
    def forward(self, X):
        X = self.effnet(X)
        output = self.out(X)
        return(output)

def model(config):
    f = globals().get(config['model']['name'])
    return f(**config['model']['params'])


if __name__ == '__main__':
    print(globals())
