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
import torchvision.models as models
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
import warnings
warnings.filterwarnings("ignore")

import math
from pathlib import Path
from torchsummary import summary
import torchvision
from PIL import Image, ImageOps

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ResNet18_fine_tune(nn.Module):
    def __init__(self):
        super(ResNet18_fine_tune, self).__init__()

        self.resnet18_model = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.resnet18_model.children())[:-1])
        
        for (name, param) in (list(self.feature_extractor.named_parameters())[:-60]):
            param.requires_grad = False
        for (name, param) in (list(self.feature_extractor.named_parameters())[-60:]):
            param.requires_grad = True

        self.classifier = nn.Sequential(nn.Linear(512, 128, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(128,16, bias=True))

    def forward(self, x):
        x=self.feature_extractor(x)
        
        x = self.classifier(torch.flatten(x, start_dim=1))
        return x
