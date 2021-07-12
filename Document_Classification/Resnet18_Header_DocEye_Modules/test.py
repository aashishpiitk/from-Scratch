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

def predict_folder_images(folder_path, resnet18_fine_tune):
   # print(folder_path)
    folder_path_predicted = create_folder('custom_images_predicted')

    for path in Path(folder_path).rglob("*.*"):
        s = str(path)
        print(path)
        img = Image.open(s)
        pred, prob = predict(s, resnet18_fine_tune)

        num = re.compile('\w{1,}')
        #folder_path = '/'.join(s.split('/')[:-1]) + '_predicted' + '/'
        img_name = num.findall(os.path.split(s)[1])[0]
        #img_name = num.findall(s.split('/')[-1])[0]

        for key in doc2label:
            if(doc2label[key] == pred):
                pred = key
                break

        
        img_path = os.path.join(folder_path_predicted, f'{img_name}_predicted_{round(prob.item(), 3)}_{pred}.png')
        img.save(str(img_path))


def predict(path, resnet18_fine_tune):
    img = Image.open(path)
    trans = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.Resize((780,600)), 
                            transforms.ToTensor()])
    
    imt = trans(img).to(device)
    imt = imt[:, :256, :]
    trans2 = transforms.Resize((227,227))
    imt = trans2(imt)
    
    #figure(figsize=(18, 16))
    #plt.imshow(imt.squeeze().detach().cpu().numpy())

    output = resnet18_fine_tune(imt.unsqueeze(0).repeat_interleave(3, dim=1))

    pred = nn.functional.softmax(output, dim=1).squeeze()

    return torch.argmax(pred), pred[torch.argmax(pred).item()]
    #print(torch.argmax(pred), pred)


if __name__ == '__main__':
    pass

