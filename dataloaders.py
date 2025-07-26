# -*- coding: utf-8 -*-

from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import shutil

def dataloaders(balanced_path, aug_tfm):
    train_path = balanced_path / 'train'
    valid_path = balanced_path / 'valid'
    train_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)

    all_files = get_image_files(balanced_path)
    labels = [parent_label(f) for f in all_files]

    train_files, valid_files, _, _ = train_test_split(
        all_files, labels, test_size=0.2, stratify=labels, random_state=42
    )

    for f in train_files:
        label = parent_label(f)
        (train_path / label).mkdir(parents=True, exist_ok=True)
        shutil.move(f, train_path / label / f.name)

    for f in valid_files:
        label = parent_label(f)
        (valid_path / label).mkdir(parents=True, exist_ok=True)
        shutil.move(f, valid_path / label / f.name)

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
        get_y=parent_label,
        item_tfms=aug_tfm,
        batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )

    return dblock.dataloaders(balanced_path, bs=64)
