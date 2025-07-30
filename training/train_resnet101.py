# -*- coding: utf-8 -*-

from fastai.vision.all import *
import timm

def train_model(dls, model_name='resnet101', pretrained=True):
    metrics = [
        accuracy,
        Precision(average='macro'),
        Recall(average='macro'),
        F1Score(average='macro'),
    ]

    learn = vision_learner(
        dls,
        model_name,
        metrics=metrics,
        loss_func=CrossEntropyLossFlat(),
        pretrained=pretrained
    ).to_fp16()

    return learn
