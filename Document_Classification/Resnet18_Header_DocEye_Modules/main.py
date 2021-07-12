
from model import ResNet18_fine_tune
from constants import doc2label
from create_and_delet_folder import create_folder, delete_folder
from model import ResNet18_fine_tune
from test import predict, predict_folder_images
from train import train
from utils import accuracy
from constants import path_to_save_weights
from dataloader import load_data, ImageDataset
from constants import dataset_path, doc2label, count, batch_size, n_cpu, n_epochs, lr, b1, b2


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description="ResNet18 Doceye Classification")
parser.add_argument("--model", default="/content/drive/MyDrive/resnet18_header_classification_with_pan_and_adhar/checkpt24.pt", help="Change the location to where the model/weights is stored")
parser.add_argument("--mode", default='test', help='Test/Train')
parser.add_argument("--load_model", default='false', help="True if you want to load the saved weights")
parser.add_argument("--save_weights", default='false', help="True if you want to save the training weights")
opt = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet18_fine_tune = ResNet18_fine_tune().to(device)
    if(opt.mode == 'test'):
        # loading the weights of model
        if(opt.load_model == 'true'):
            checkpoint = torch.load(opt.model, map_location = device)
        
        if(opt.load_model == 'true'):
            resnet18_fine_tune.load_state_dict(checkpoint['resnet18_fine_tune'])

        resnet18_fine_tune.eval()

        folder_name = "custom_images"
        custom_img_folder_path = os.path.join(os.getcwd(), folder_name)
        predict_folder_images(custom_img_folder_path, resnet18_fine_tune)

    if(opt.mode == 'train'):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet18_fine_tune.parameters()), lr=lr, betas=(b1, b2))
        criterion = nn.CrossEntropyLoss()
        train_dataloader, test_dataloader = load_data(dataset_path)
        
        train(opt.save_weights, device, resnet18_fine_tune, train_dataloader, test_dataloader, optimizer, criterion)
        

if __name__ == "__main__":
    main()
