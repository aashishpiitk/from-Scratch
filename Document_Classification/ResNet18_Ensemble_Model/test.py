import re
import PIL
from matplotlib.pyplot import figure
import os
from os import path
import shutil
from constants import doc2label
from PIL import Image, ImageOps
from pathlib import Path
from create_and_delet_folder import create_folder, delete_folder
from model import ResNet18_fine_tune
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from constants import label2doc, load_weights, extract_page_sections, folder_path_to_save_predicted_images, folder_path_to_test_images

def predict_folder_images(model_components, device, debug=False):
    
    for path in Path(folder_path_to_test_images).rglob("*.*"):
        s = str(path)
        
        if(debug):
            print(path)
        
        img = Image.open(s)
        doc_class, doc_pred = make_prediction(model_components, s, device, debug)

        num = re.compile('\w{1,}')
        img_name = num.findall(os.path.split(s)[1])[0]
        
        img_path = os.path.join(folder_path_to_save_predicted_images, f'{img_name}_predicted_{round(doc_pred, 3)}_{doc_class}.png')
        img.save(str(img_path))

def make_prediction(model_components, path_to_image, device, debug=False):
    load_weights(model_components, device, isEnsemble=True)
    
    meta_classifier = model_components['meta_classifier']
    resnet18_header = model_components['resnet18_header']
    resnet18_footer = model_components['resnet18_footer']
    resnet18_right_half = model_components['resnet18_right_half']
    resnet18_left_half = model_components['resnet18_left_half']
    resnet18_holistic = model_components['resnet18_holistic']

    page_sections = extract_page_sections(path_to_image)

    header = page_sections['header']
    footer = page_sections['footer']
    holistic = page_sections['holistic']
    right_half = page_sections['right_half']
    left_half = page_sections['left_half']

    left_pred = resnet18_left_half(left_half).detach()
    right_pred = resnet18_right_half(right_half).detach()
    header_pred = resnet18_header(header).detach()
    footer_pred = resnet18_footer(footer).detach()
    holistic_prediction = resnet18_holistic(holistic).detach()
    concatenated_pred = torch.cat((left_pred, right_pred, header_pred, footer_pred, holistic_prediction), dim=1)
    final_pred = meta_classifier(concatenated_pred)

    # apply softmax to get the probabilities
    final_pred = nn.Softmax(final_pred, dim=1).squeeze()
    doc_class = label2doc[torch.argmax(final_pred, dim=1)]
    doc_pred = final_pred[torch.argmax(final_pred, dim=1)].item()

    if(debug):
        print(doc_class, doc_pred)
    
    return doc_class, doc_pred
    # save the predicted image to disk along with the class name and prob attached to its name


