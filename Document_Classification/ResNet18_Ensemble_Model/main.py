from ResNet18_Ensemble_Model.train import train_ensemble
from model import ResNet18_fine_tune
from constants import doc2label
from create_and_delet_folder import create_folder, delete_folder
from model import ResNet18_fine_tune, MetaClassifier
from test import predict, predict_folder_images
from train import train
from utils import accuracy
from constants import path_to_save_weights
from dataloader import load_data, ImageDataset
from constants import dataset_path, doc2label, count, batch_size, n_cpu, n_epochs, lr, b1, b2
from utils import create_required_folders, create_folder, create_features_for_page_sections
from train import train_section

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import argparse, os
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Resize, Compose
import re
import PIL
from matplotlib.pyplot import figure
import os
from os import path
import shutil
from PIL import Image, ImageOps
from pathlib import Path
from sklearn.model_selection import train_test_split
import torchvision

parser = argparse.ArgumentParser(description="ResNet18 Doceye Classification")
parser.add_argument("--model", default="/content/drive/MyDrive/resnet18_header_classification_with_pan_and_adhar/checkpt24.pt", help="Change the location to where the model/weights is stored")
parser.add_argument("--mode", default='test', help='test/train/create_folders')
parser.add_argument("--load_model", default = False, help="True if you want to load the saved weights")
parser.add_argument("--save_weights", default = False, help="True if you want to save the training weights")

opt = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    create_required_folders()
    page_sections = {'train_holistic', 'train_header', 'train_footer', 'train_right_half', 'train_left_half'}

    if(opt.mode == 'create_folders'):
        pass

    if(opt.mode == 'test'):

        model_components = {
            'resnet18_header' : ResNet18_fine_tune().to(device),
            'resnet18_footer' : ResNet18_fine_tune().to(device),
            'resnet18_right_half' : ResNet18_fine_tune().to(device),
            'resnet18_left_half' : ResNet18_fine_tune().to(device),
            'resnet18_holistic' : ResNet18_fine_tune().to(device),
            'meta_classifier' : MetaClassifier().to(device)
        }

        predict_folder_images(model_components, device, debug=False)

    if(opt.mode in page_sections):
        resnet18_fine_tune = ResNet18_fine_tune()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet18_fine_tune.parameters()), lr=lr, betas=(b1, b2))
        criterion = nn.CrossEntropyLoss()
        page_section = (opt.mode).split('_')[1:]
        train_dataloader, test_dataloader = load_data(dataset_path, page_section)
        
        train_section(page_section, opt.save_weights, device, resnet18_fine_tune, train_dataloader, test_dataloader, optimizer, criterion)

    if(opt.mode == 'train_ensemble'):
        train_dataloader, test_dataloader = load_data(dataset_path, 'ensemble')
        dataloader = (train_dataloader, test_dataloader)
        
        meta_classifier = MetaClassifier().to(device)
        model_components = {
            'resnet18_header' : ResNet18_fine_tune().to(device),
            'resnet18_footer' : ResNet18_fine_tune().to(device),
            'resnet18_right_half' : ResNet18_fine_tune().to(device),
            'resnet18_left_half' : ResNet18_fine_tune().to(device),
            'resnet18_holistic' : ResNet18_fine_tune().to(device),
            'meta_classifier' : meta_classifier,
            'optimizer' : torch.optim.Adam(filter(lambda p: p.requires_grad, meta_classifier.parameters()), lr=lr, betas=(b1, b2)).to(device),
            'criterion' : nn.CrossEntropyLoss().to(device)
        }
        
        train_ensemble(model_components, dataloader, opt.save_weights, device)#search for the notebook containing the training loop

if __name__ == "__main__":
    main()
