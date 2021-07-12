import torch
import torch.nn as nn
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from constants import path_to_save_weights 

def accuracy(x, targets):
  probs = nn.functional.softmax(x)
  labels = torch.argmax(probs, dim=1)

  count=0
  for label, target in zip(labels, targets):
    if(label.item() == target.item()):
      count+=1
  
  return count/labels.shape[0]

def create_folder(folderpath):
    '''Creates a new folder in the current directory named as folder_name
    
        Parameters:
            folder_name (string) : name of the folder to create
    '''
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        if os.path.exists(folderpath):
            print('Folder Created', folderpath)
    
def create_required_folders():
    '''Creates the default folders in the current directory'''

    create_folder(os.path.join(os.getcwd(),'custom_images_predicted'))
    create_folder(os.path.join(os.getcwd(),'custom_images'))
    create_folder(os.path.join(os.getcwd(),'images_dataset'))
    create_folder(os.path.join(os.getcwd(),'saved_state_dicts'))
    create_folder(os.path.join(os.getcwd(),'saved_state_dicts', 'footer'))
    create_folder(os.path.join(os.getcwd(),'saved_state_dicts', 'header'))
    create_folder(os.path.join(os.getcwd(),'saved_state_dicts', 'holistic'))
    create_folder(os.path.join(os.getcwd(),'saved_state_dicts', 'left_half'))
    create_folder(os.path.join(os.getcwd(),'saved_state_dicts', 'right_half'))

def load_weights(model_components, device, isEnsemble=False):
    resnet18_header = model_components['resnet18_header']
    resnet18_footer = model_components['resnet18_footer']
    resnet18_right_half = model_components['resnet18_right_half']
    resnet18_left_half = model_components['resnet18_left_half']
    resnet18_holistic = model_components['resnet18_holistic']

    header = path_to_save_weights['header']
    footer = path_to_save_weights['footer']
    right_half = path_to_save_weights['right_half']
    left_half = path_to_save_weights['left_half']
    holistic = path_to_save_weights['holistic']

    checkpoint = torch.load(f"{header}", map_location=device)
    resnet18_header.load_state_dict(checkpoint['resnet18_fine_tune'])

    checkpoint = torch.load(f"{footer}", map_location=device)
    resnet18_footer.load_state_dict(checkpoint['resnet18_fine_tune'])

    checkpoint = torch.load(f"{holistic}", map_location=device)
    resnet18_holistic.load_state_dict(checkpoint['resnet18_fine_tune'])

    checkpoint = torch.load(f"{right_half}", map_location=device)
    resnet18_right_half.load_state_dict(checkpoint['resnet18_fine_tune'])

    checkpoint = torch.load(f"{left_half}", map_location=device)
    resnet18_left_half.load_state_dict(checkpoint['resnet18_fine_tune'])

    if(isEnsemble):
        meta_classifier = model_components['meta_classifier']
        ensemble = path_to_save_weights['ensemble']
        checkpoint = torch.load(f"{ensemble}", map_location=device)
        meta_classifier.load_state_dict(checkpoint['meta_classifier'])

def extract_page_sections(path_to_image, device):
    img = Image.open(path_to_image)
    trans = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.Resize((780,600)), 
                            transforms.ToTensor()])
    trans2 = transforms.Resize((227,227))
                                        
    imt = trans(img).to(device)

    image_sections = {
        'header' : trans2(imt[:, :256, :]).unsqueeze(0).repeat_interleave(3, dim=1),
        'footer' : trans2(imt[:, -256:, :]).unsqueeze(0).repeat_interleave(3, dim=1),
        'holistic' : trans2(imt).unsqueeze(0).repeat_interleave(3, dim=1),
        'left_half' : trans2(imt[:, 100:-100, :300]).unsqueeze(0).repeat_interleave(3, dim=1),
        'right_half' : trans2(imt[:, 100:-100, -300:]).unsqueeze(0).repeat_interleave(3, dim=1)
    }
    return image_sections
    
