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
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
import math
from pathlib import Path
from torchsummary import summary
import torchvision
from PIL import Image, ImageOps
random.seed(42)

from utils import accuracy, load_weights, extract_page_sections
from constants import path_to_save_weights, label2doc

def train_section(page_section, save_weights=False, device, resnet18_fine_tune, train_dataloader, test_dataloader, optimizer, criterion):

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
            batch_size = imgs.shape[0]
            optimizer.zero_grad()


            resnet18_fine_tune.train();

            for key in imgs:
                imgs[key] = imgs[key].to(device)

            holistic_img = imgs[f'{page_section}']
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
            batch_size = imgs.shape[0]
            
            resnet18_fine_tune.eval();
            
            for key in imgs:
                imgs[key] = imgs[key].to(device)

            holistic_img = imgs[f'{page_section}']
            targets = imgs['targets']
            prediction = resnet18_fine_tune(holistic_img.to(device))

            loss_calc = criterion(prediction.to(device), targets.to(device))
            loss += loss_calc.item()
            acc += accuracy(prediction.to(device), targets.to(device))
            tqdm_bar.set_postfix(loss=loss/(batch_idx+1), acc = acc/(batch_idx+1))

        test_losses.append(loss/len(test_dataloader)) ## divided by total number of batches

        path_to_save_weight = path_to_save_weights[f'{page_section}']
        if(save_weights):
            torch.save({
                'resnet18_fine_tune' : resnet18_fine_tune.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss' : train_losses[-1],
                'test_loss' : test_losses[-1],
                'epoch': epoch+1,
                'accuracy' : acc/len(tqdm_bar),
                }, os.path.join(path_to_save_weight, f'checkpt{epoch+1}.pt'))

def train_ensemble(model_components, dataloader, save_weights, device):
    
    optimizer = model_components['optimizer']
    criterion = model_components['criterion']
    
    load_weights(model_components, device)
    
    meta_classifier = model_components['meta_classifier']
    resnet18_header = model_components['resnet18_header']
    resnet18_footer = model_components['resnet18_footer']
    resnet18_right_half = model_components['resnet18_right_half']
    resnet18_left_half = model_components['resnet18_left_half']
    resnet18_holistic = model_components['resnet18_holistic']

    train_dataloader, test_dataloader = dataloader
    
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
            batch_size = imgs.shape[0]
            optimizer.zero_grad()

            meta_classifier.train();

            targets = imgs['targets'].to(device)
            header = imgs['header'].to(device)
            footer = imgs['footer'].to(device)
            holistic = imgs['holistic'].to(device)
            right_half = imgs['right_half'].to(device)
            left_half = imgs['left_half'].to(device)

            left_pred = resnet18_left_half(left_half).detach()
            right_pred = resnet18_right_half(right_half).detach()
            header_pred = resnet18_header(header).detach()
            footer_pred = resnet18_footer(footer).detach()
            holistic_prediction = resnet18_holistic(holistic).detach()
            concatenated_pred = torch.cat((left_pred, right_pred, header_pred, footer_pred, holistic_prediction), dim=1)
            
            final_pred = meta_classifier(concatenated_pred)

            loss_calc = criterion(final_pred.to(device), targets.to(device))
            loss_calc.backward()
            optimizer.step()

            loss += loss_calc.item()
            acc += accuracy(final_pred.to(device), targets.to(device))
            tqdm_bar.set_postfix(loss=loss/(batch_idx+1), acc = acc/(batch_idx+1))

        train_losses.append(loss/len(train_dataloader)) ## divided by total number of batches

        ## Testing loop
        loss = 0
        acc = 0
        tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))
        for batch_idx, imgs in enumerate(tqdm_bar):
            batch_size = imgs.shape[0]
            optimizer.zero_grad()

            meta_classifier.train();

            targets = imgs['targets'].to(device)
            header = imgs['header'].to(device)
            footer = imgs['footer'].to(device)
            holistic = imgs['holistic'].to(device)
            right_half = imgs['right_half'].to(device)
            left_half = imgs['left_half'].to(device)

            left_pred = resnet18_left_half(left_half).detach()
            right_pred = resnet18_right_half(right_half).detach()
            header_pred = resnet18_header(header).detach()
            footer_pred = resnet18_footer(footer).detach()
            holistic_prediction = resnet18_holistic(holistic).detach()
            concatenated_pred = torch.cat((left_pred, right_pred, header_pred, footer_pred, holistic_prediction), dim=1)
            
            final_pred = meta_classifier(concatenated_pred)

            loss_calc = criterion(final_pred.to(device), targets.to(device))

            loss += loss_calc.item()
            acc += accuracy(final_pred.to(device), targets.to(device))
            tqdm_bar.set_postfix(loss=loss/(batch_idx+1), acc = acc/(batch_idx+1))

        test_losses.append(loss/len(test_dataloader)) ## divided by total number of batches

        path_to_save_weight = path_to_save_weights['ensemble']
        if(True):
            torch.save({
                
                'meta_classifier' : meta_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss' : train_losses[-1],
                'test_loss' : test_losses[-1],
                'epoch': epoch+1,
                'accuracy' : acc/len(tqdm_bar),
                }, os.path.join(path_to_save_weight, f'checkpt{epoch+1}.pt'))





