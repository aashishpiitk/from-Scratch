import numpy as np
import pandas as pd
import os, math, sys
import glob, itertools
import argparse, random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split

random.seed(42)
import math
from pathlib import Path
from torchsummary import summary
import torchvision
from PIL import Image, ImageOps

from utils import accuracy
from constants import path_to_save_weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(save_weights, device, resnet18_fine_tune, train_dataloader, test_dataloader, optimizer, criterion):

    n_epochs = 24

    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    test_counter = [idx*len(test_dataloader.dataset) for idx in range(1, n_epochs+1)]
    train_counter = [idx*len(train_dataloader.dataset) for idx in range(1, n_epochs+1)]

    for epoch in range(n_epochs):

        # Training loop
        loss = 0
        acc = 0
        tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))

        for batch_idx, imgs in enumerate(tqdm_bar):

            optimizer.zero_grad()


            resnet18_fine_tune.train();

            for key in imgs:
                imgs[key] = imgs[key].to(device)

            holistic_img = imgs['header']
            targets = imgs['targets']

            prediction = resnet18_fine_tune(holistic_img.to(device))

            loss_calc = criterion(prediction.to(device), targets.to(device))

            loss_calc.backward()
            optimizer.step()

            loss += loss_calc.item()
            acc += accuracy(prediction.to(device), targets.to(device))
            tqdm_bar.set_postfix(loss=loss/(batch_idx+1), acc = acc/(batch_idx+1))

        train_losses.append(loss/len(train_dataloader)) ## divided by total number of batches


        ## Testing loop
        loss = 0
        acc = 0
        tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))

        for batch_idx, imgs in enumerate(tqdm_bar):

            resnet18_fine_tune.eval();
            
            for key in imgs:
                imgs[key] = imgs[key].to(device)

            holistic_img = imgs['header']
            targets = imgs['targets']

            prediction = resnet18_fine_tune(holistic_img.to(device))

            loss_calc = criterion(prediction.to(device), targets.to(device))

            loss += loss_calc.item()
            acc += accuracy(prediction.to(device), targets.to(device))
            tqdm_bar.set_postfix(loss=loss/(batch_idx+1), acc = acc/(batch_idx+1))

        test_losses.append(loss/len(test_dataloader)) ## divided by total number of batches

        if(save_weights):
            torch.save({
                
                'resnet18_fine_tune' : resnet18_fine_tune.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss' : train_losses[-1],
                'test_loss' : test_losses[-1],
                'epoch': epoch+1,
                }, f"{path_to_save_weights}/checkpt{epoch+1}.pt")