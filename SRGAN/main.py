import argparse, os
from model import GeneratorResNet, Discriminator, ResidualBlock, FeatureExtractor
import torch.nn as nn
from utils import resize_snip_super_resolve_patch
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image, ImageOps
import torch
import random
import torchvision
from pathlib import Path
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from constants import lr, b1, b2, dataset_path, channels, hr_shape, low_resolution_images_path
from train import train
from dataloader import load_data
from utils import load_state_dict, create_folder, create_required_folders

parser = argparse.ArgumentParser(description="SRGAN for 2x, 4x, 8x, ...")
#parser.add_argument("--img_location", type=str, help="Add the location in a string format")
parser.add_argument("--checkpoint", default=f"{os.path.join(os.getcwd(), 'saved_state_dicts/checkpt30.pt')}", help="Change the location to where the model weights are stored")
parser.add_argument("--save_folder", type=str, default=".", help="location of folder where to save the output image in string format")
parser.add_argument('--mode', default='test', type=str, help="Mode in which to run the model")
parser.add_argument('--load_state_dict', type=bool, default=False, help="Whether you want to load the state dict for the model and optimizers")
parser.add_argument('--SR_factor', type=int, default=2, help="Give 2 for 2x super resolution, 4 for 4x, ...")
parser.add_argument('--save_weights', type=bool, default=False, help="Whether to save the state dicts of model and optimizer during training")
parser.add_argument('--debug', type=bool, default=False, help="Turn on if you want to print the debug statemens in the code")

opt = parser.parse_args()

def main():
    
	# create the required default folders if they don't exist
	create_required_folders()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if(opt.mode == 'test'):
			
		# generator
		generator = GeneratorResNet().to(device)
		# loading the weights of model
		if(opt.load_state_dict):
			checkpoint = torch.load(opt.checkpoint, map_location = device)
			generator.load_state_dict(checkpoint['generator_state_dict'])
		generator.eval()
		# feature_extractor = FeatureExtractor().to(device)
		# feature_extractor.eval()
		

		for path in Path(low_resolution_images_path).rglob('*.*'):
			img_location = str(path)
			resize_snip_super_resolve_patch(img_location, generator, SR_factor=opt.SR_factor)
	
	if(opt.mode == 'train'):
		generator = GeneratorResNet(SR_factor = opt.SR_factor).to(device)
		discriminator = Discriminator(input_shape=(channels, *hr_shape), SR_factor = opt.SR_factor).to(device)
		feature_extractor = FeatureExtractor().to(device)
		feature_extractor.eval()

		# Losses
		criterion_GAN = torch.nn.MSELoss().to(device)
		criterion_content = torch.nn.L1Loss().to(device)
	
		# Optimizers
		optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
		optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    	
		# load the saved weights
		if(opt.load_state_dict):
			checkpoint = torch.load(opt.checkpoint, map_location = device)
			load_state_dict(generator, discriminator, optimizer_G, optimizer_D, checkpoint)	

		model_components = {
			'generator' : generator,
			'discriminator' : discriminator,
			'feature_extractor' : feature_extractor,
			'criterion_GAN' : criterion_GAN,
			'criterion_content' : criterion_content,
			'optimizer_G' : optimizer_G,
			'optimizer_D' : optimizer_D
		}

		# prepare dataloader from dataset path
		dataloader = load_data(dataset_path, SR_factor = opt.SR_factor)
		
		#train
		train(dataloader, model_components, opt.save_weights, debug=opt.debug)


if __name__ == "__main__":
	main()


