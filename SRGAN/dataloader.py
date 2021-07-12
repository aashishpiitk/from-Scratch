from constants import dataset_path, batch_size, n_cpu, hr_shape
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
import random
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
random.seed(42)
import math
from pathlib import Path

class ImageDataset(Dataset):
    '''Takes an image of a page, crops a region of size hr_shape from the center of page
        and then creates a low resolution images by resizing the image to half the height and width'''
    def __init__(self, files, hr_shape, SR_factor=2):
        hr_height, hr_width = hr_shape
        self.lr_transform = transforms.Compose(
            [
                transforms.CenterCrop((hr_height, hr_width)),
                transforms.Resize((hr_height//SR_factor, hr_width//SR_factor)),
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.CenterCrop((hr_height, hr_width)),
                transforms.ToTensor(),
            ]
        )
        self.files = files
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)


def load_data(dataset_path, SR_factor = 2):
    '''Extracts the .tif files from a folder, and makes train and test dataloader with seed=42'''
    train_path = []
    for path in Path(dataset_path).rglob("*.tif"):
        train_path.append(path)
    
    train_paths, test_paths = train_test_split(train_path, test_size=0.1, random_state=42)
    train_dataloader = DataLoader(ImageDataset(train_paths, hr_shape=hr_shape, SR_factor = SR_factor), batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    test_dataloader = DataLoader(ImageDataset(test_paths, hr_shape=hr_shape, SR_factor = SR_factor), batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    return train_dataloader, test_dataloader