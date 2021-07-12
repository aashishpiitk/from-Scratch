import numpy as np
import pandas as pd
import os, math, sys
import glob, itertools
import argparse, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
random.seed(42)
import math

from utils import save_state_dict
from constants import batch_size, save_state_dict_path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train(dataloader, model_components, save_weights, n_epochs=25, debug=False):
    
    optimizer_G = model_components['optimizer_G']
    optimizer_D = model_components['optimizer_D']
    generator = model_components['generator']
    discriminator = model_components['discriminator']
    feature_extractor = model_components['feature_extractor']
    criterion_GAN = model_components['criterion_GAN']
    criterion_content = model_components['criterion_content']

    train_dataloader, test_dataloader = dataloader
    
    if(debug):
        print(len(train_dataloader), len(test_dataloader))
    
    train_gen_losses, train_disc_losses, train_counter = [], [], []
    test_gen_losses, test_disc_losses = [], []
    test_counter = [idx*len(train_dataloader.dataset) for idx in range(1, n_epochs+1)]

    for epoch in range(n_epochs):

        ### Training
        gen_loss, disc_loss = 0, 0
        tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))
        for batch_idx, imgs in enumerate(tqdm_bar):
            batch_size = imgs['lr'].shape[0]
            
            generator.train(); discriminator.train()
            # Configure model input
            imgs_lr = imgs["lr"]
            imgs_hr = imgs["hr"]

            # Adversarial ground truths
            valid = torch.ones((imgs_lr.size(0), *discriminator.output_shape), requires_grad=False)
            fake = torch.zeros((imgs_lr.size(0), *discriminator.output_shape), requires_grad=False)
            
            ### Train Generator
            optimizer_G.zero_grad()
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss
            # converting the 1 channel input image to 3 channel input image, with the last two channels containing zero
            gen_hr_3 = gen_hr.repeat_interleave(3, dim=1)
            # first dimension is for batch_size, second for channels, and the last two for feature map
            gen_hr_3[:,1:3,:,:]=0 
            # VGG feature exractor accepts images with three channels only
            imgs_hr_3 = imgs_hr.repeat_interleave(3, dim=1)
            imgs_hr_3[:,1:3,:,:] = 0

            
            gen_features = feature_extractor(gen_hr_3)
            real_features = feature_extractor(imgs_hr_3)
            loss_content = criterion_content(gen_features, real_features.detach())##detached because no backprop in VGG
            
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            optimizer_G.step()

            ### Train Discriminator
            optimizer_D.zero_grad()
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            gen_loss += loss_G.item()
            train_gen_losses.append(loss_G.item())
            disc_loss += loss_D.item()
            train_disc_losses.append(loss_D.item())
            train_counter.append(batch_idx*batch_size + imgs_lr.size(0) + epoch*len(train_dataloader.dataset))
            tqdm_bar.set_postfix(gen_loss = gen_loss/(batch_idx+1), disc_loss = disc_loss/(batch_idx+1))
            if(debug):
                print('Training Batch', batch_idx, 'Processed')
                print('GenLoss', gen_loss/(batch_idx+1))
                print('DiscLoss', disc_loss/(batch_idx+1))

        # Testing
        gen_loss, disc_loss = 0, 0
        tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))
        for batch_idx, imgs in enumerate(tqdm_bar):
            batch_size = imgs.shape[0]

            generator.eval(); discriminator.eval()
            # Configure model input
            imgs_lr = imgs["lr"]
            imgs_hr = imgs["hr"]
            # Adversarial ground truths
            valid = torch.ones((imgs_lr.size(0), *discriminator.output_shape), requires_grad=False)
            fake = torch.zeros((imgs_lr.size(0), *discriminator.output_shape), requires_grad=False)
            
            ### Eval Generator
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            
            # Content loss
            # converting the 1 channel input image to 3 channel input image, with the last two channels containing zero
            gen_hr_3 = gen_hr.repeat_interleave(3, dim=1)
            ## first dimension is for batch_size, second for channesls, and the last two for feature map
            gen_hr_3[:,1:3,:,:]=0 
            imgs_hr_3 = imgs_hr.repeat_interleave(3, dim=1)
            imgs_hr_3[:,1:3,:,:] = 0
            
            gen_features = feature_extractor(gen_hr_3)
            real_features = feature_extractor(imgs_hr_3)
            loss_content = criterion_content(gen_features, real_features.detach())## real_features is detached because VGG which we are using here is 
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            ### Eval Discriminator
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            gen_loss += loss_G.item()
            disc_loss += loss_D.item()
            tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))

            if(debug):
                print('Testing Batch', batch_idx, 'Processed')
                print('GenLoss', gen_loss/(batch_idx+1))
                print('DiscLoss', disc_loss/(batch_idx+1))

        test_gen_losses.append(gen_loss/len(test_dataloader))
        test_disc_losses.append(disc_loss/len(test_dataloader))
        

        if(save_weights and ((epoch+1)%5 == 0 or epoch==0)):
            save_state_dict(generator, discriminator, optimizer_G, optimizer_D, epoch, save_state_dict_path)
            
            
            