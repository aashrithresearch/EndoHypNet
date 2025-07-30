# -*- coding: utf-8 -*-

import albumentations as A
import numpy as np
from fastai.vision.all import *

class AlbumentationsTransform(Transform):
    def __init__(self, aug): self.aug = aug

    def encodes(self, img: PILImage):
        np_img = np.array(img)
        aug_img = self.aug(image=np_img)['image']
        return PILImage.create(aug_img)

def get_albumentations():
    import albumentations as A
    return A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(p=0.2),
        A.ElasticTransform(p=0.2),
    ])
