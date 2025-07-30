# EndoHypNet: Histopathology-Aware Classifier for Early Endometrial Hyperplasia Diagnosis using Deep Learning
This repository provides a full pipeline for training deep learning models (InceptionV3, ResNet50, MobileNetV5) on histopathological images to classify endometrial tissue as **Normal Endometrium** or **Endometrial Hyperplasia**
Built using `Fastai`, `Timm`, and `albumentations`, the pipeline also supports explainability using SHAP and Integrated Gradients

## Dataset
This project uses the dataset from:

**Zhang et al., 2018.**  
**A histopathological image dataset for endometrial disease diagnosis.**  
Available on [Figshare](https://figshare.com/articles/dataset/A_histopathological_image_dataset_for_endometrial_disease_diagnosis/7306361)
Includes histopathological data of the following: `Normal Endometrium`, `Endometrial Hyperplasia`, `Endometroid Adenocarcinoma`, `Endometrial Polyp`

## Repo Structure
The repository is structured as below:
`data/`: Raw dataset folder, only uses classes Endometrial Hyperplasia and Normal Endometrium from dataset

`models/`: Exported .pkl models, includes inception_v3.pkl, resnet50.pkl, and mobilenet_v5.pkl

`scripts/`: Scripts necessary for running notebooks. Includes: balance.py (oversampling/undersampling functions), augmentations.py (albumentations integration for FastAI), dataloaders.py (augmented dataloaders), and explain.py (explainability module with SHAP and Integrated Gradients)

`training/`: Scripts necessary for training models on dataset, includes scripts for inception_v3, resnet50, and mobilenet_v5

`notebooks/`: Notebooks that include all main-training pipelines for each model. Highest performing model: inception_v3 (refer to metrics)

`metrics/`: Metrics for all models, includes confusion matrix and classification report on the validation set. Classification report includes precision, recall, and f1-scores specific to each class. 

## Set-up Instructions
### 1. Clone repo
```bash
git clone https://github.com/aashrithresearch/EndoHypNet.git
cd EndoHypNet

### 2. Install dependencies
```bash
pip install -r requirements.txt
