from torch.utils.data import Dataset
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import albumentations as A
from albumentations.augmentations.bbox_utils import convert_bbox_from_albumentations
import torch
"""
This python file holds the datasets of the project.
Copy and paste your datasets to this file.
The gen_dataloader file will import this file and using the config
name import the correct datast.
"""

class OD_SIIM(Dataset):
    def __init__(self, image_ids, df, transforms=None):
        super().__init__()
        # image_ids
        self.image_ids = image_ids
        # random sample data
        self.df = df
        # augmentations
        self.transforms = transforms
    
    def __len__(self) -> int:
        return(len(self.image_ids))
    
    @staticmethod
    def dicom2array(path: str, voi_lut=True, fix_monochrome=True):
        dicom = pydicom.read_file(path)
        # VOI LUT (if available by DICOM device) is used to
        # transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
        # depending on this value, X-ray may look inverted - fix that:
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        return data.astype(np.float32)
    
    def load_bbox_labels(self, image_id, shape):
        rows, cols = shape
        records = self.df[self.df['id'] == image_id]
        new_boxes = []
        for boxes in records.boxes.values:
            if boxes:
                for box in boxes:
                    frac_box = np.clip((box['x']/cols, box['y']/rows, (box['x']+box['width'])/cols,
                                      (box['y']+box['height'])/rows), 0, 1)
                    converted = convert_bbox_from_albumentations(bbox=frac_box, 
                                                 target_format='pascal_voc', 
                                                 rows=rows, cols=cols, 
                                                 check_validity=True)
                    new_boxes.append(converted)
        labels = [records['integer_label'].values[0]] * len(boxes)
        return(new_boxes, labels)
        
    def __getitem__(self, idx: int):
        # retrieve idx data
        image_id = self.image_ids[idx]
        # get path
        image_path = self.df[self.df['id'] == image_id].file_path.values[0]
        # get image
        image = self.dicom2array(image_path)
        # get boxes and labels
        boxes, labels = self.load_bbox_labels(image_id, image.shape)
        if self.transforms:
            tform = self.transforms(image=image, 
                              bboxes=boxes, 
                              labels=labels)
            image = tform['image']
            target = {'boxes': torch.tensor(tform['bboxes']), 
                      'labels': torch.tensor(tform['labels'])}
            # 1 Channel vs 3 Channel?
            image = np.dstack((image, image, image))
            return(torch.tensor(image).permute(2, 0, 1), target, image_id)
            
        return image, boxes, image_id

class IC_SIIM(Dataset):
    def __init__(self, image_ids, df, transforms=None):
        super().__init__()
        # image_ids
        self.image_ids = image_ids
        # random sample data
        self.df = df
        # augmentations
        self.transforms = transforms
    
    def __len__(self) -> int:
        return(len(self.image_ids))
    
    @staticmethod
    def dicom2array(path: str, voi_lut=True, fix_monochrome=True):
        dicom = pydicom.read_file(path)
        # VOI LUT (if available by DICOM device) is used to
        # transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
        # depending on this value, X-ray may look inverted - fix that:
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        return data.astype(np.float32)
    
    def load_labels(self, image_id):
        # find row
        records = self.df[self.df['id'] == image_id]
        # mapping
        mapping = {0: 'Negative for Pneumonia',
                    1: 'Typical Appearance',
                    2: 'Indeterminate Appearance',
                    3: 'Atypical Appearance'}
        # get label
        labels = records[[mapping[0], mapping[1], mapping[2], mapping[3]]].values[0]
        #label = records['integer_label'].values[0] # only one label per image
        return(labels)
    
    def __getitem__(self, idx: int):
        # retrieve idx data
        image_id = self.image_ids[idx]
        # get path
        image_path = self.df[self.df['id'] == image_id].file_path.values[0]
        # get image
        image = self.dicom2array(image_path)
        # get labels
        labels = self.load_labels(image_id)
        # Augments
        if self.transforms:
            t = self.transforms(**{'image': image})
            image = t['image']
        image = np.dstack((image, image, image))
        return (torch.tensor(image).permute(2, 0, 1), torch.tensor(labels), image_id)

def dataset(name, df, train_ids, transform):
    d = globals().get(name)
    return(d(train_ids, df, transform))
