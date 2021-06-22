from torch.utils.data import Dataset
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
        row, col = shape
        records = self.df[self.df['id'] == image_id]
        boxes = []
        for box in records.boxes.values:
            if box:
                for b in box:
                    to_append = np.clip([b['x']/col, b['y']/row, (b['x']+b['width'])/col,
                                         (b['y']+b['height'])/row], 0, 1.0)
                    temp = A.convert_bbox_from_albumentations(to_append,
                                                              'pascal_voc',
                                                              rows=row, cols=col)
                    boxes.append(temp)

        labels = [records['integer_label'].values[0]] * len(boxes)
        return(boxes, labels)

    def __getitem__(self, idx: int):
        # retrieve idx data
        image_id = self.image_ids[idx]
        # get path
        image_path = self.df[self.df['id'] == image_id].file_path.values[0]
        # get image
        image = self.dicom2array(image_path)
        # get boxes and labels
        boxes, labels = self.load_bbox_labels(image_id, image.shape)
        # target
        target = {
            'boxes': boxes,
            'labels': labels
        }
        # sample
        sample = {
            'image': image,
            'bboxes': boxes,
            'labels': labels
        }
        # Augments
        if self.transforms:
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])
            if target["boxes"].shape[0] != 0:
                target['boxes'] = torch.stack(tuple(map(torch.tensor,
                                                 zip(*sample['bboxes'])))).permute(1, 0)
            target['labels'] = torch.tensor(sample['labels'])
        return image, target, image_id


def dataset(name, df, train_ids, transform):
    d = globals().get(name)
    return(d(**_))
