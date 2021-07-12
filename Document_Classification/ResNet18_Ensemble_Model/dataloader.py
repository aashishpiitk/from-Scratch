from constants import dataset_path, doc2label, count, batch_size, n_cpu, n_epochs
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image, ImageOps

class ImageDataset(Dataset):
    def __init__(self, files, page_section):

        self.files = files
        self.page_section = page_section
        self.trans = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.Resize((780,600)), 
                                    transforms.ToTensor(),
                                    
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        self.trans2 = transforms.Resize((227,227))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)][0])
        target = self.files[index % len(self.files)][1]
        
        output_dict = {
            'targets' : torch.tensor(target),
        }

        if(self.page_section == 'header'):
            output_dict['header'] = self.create_header(img)
        if(self.page_section == 'footer'):
            output_dict['footer'] = self.create_footer(img)
        if(self.page_section == 'right_half'):
            output_dict['right_half'] = self.create_right_half(img)
        if(self.page_section == 'left_half'):
            output_dict['left_half'] = self.create_left_half(img)
        if(self.page_section == 'holistic'):
            output_dict['holistic'] = self.create_holistic(img)
        if(self.page_section == 'ensemble'):
            output_dict['header'] = self.create_header(img)
            output_dict['footer'] = self.create_footer(img)
            output_dict['right_half'] = self.create_right_half(img)
            output_dict['left_half'] = self.create_left_half(img)
            output_dict['holistic'] = self.create_holistic(img)
        return output_dict
    
    def __len__(self):
        return len(self.files)
    
    def create_header(self, x):
        x = self.trans(x)
        x = x[:][:, :256, :]
        x = x.repeat_interleave(3, dim=0)
        return self.trans2(x)

    def create_right_half(self, x):
        x = self.trans(x)
        x = x[:][:, 100:-100, -300:]
        x = x.repeat_interleave(3, dim=0)
        return self.trans2(x)
    
    def create_left_half(self, x):
        x = self.trans(x)
        x = x[:][:, 100:-100, :300]
        x = x.repeat_interleave(3, dim=0)
        return self.trans2(x)

    def create_footer(self, x):
        x = self.trans(x)
        x = x[:][:, -256:, :]
        x = x.repeat_interleave(3, dim=0)
        return self.trans2(x)

    def create_holistic(self, x):
        x = self.trans(x)
        x = x.repeat_interleave(3, dim=0)
        return self.normalize(x)

def load_data(dataset_path, page_section='holistic'):
    train_path = []
    for path in Path(dataset_path).rglob('*.png'):
        target = str(str(path).split('/')[-2])
        count[target]+=1
        if(count[target] > 75):
            continue
        train_path.append((path, doc2label[target]))

    train_paths, test_paths = train_test_split(train_path, test_size=0.1, random_state=42)
    train_paths = train_paths[:len(train_paths)]
    test_paths = test_paths[:len(test_paths)]

    train_dataloader = DataLoader(ImageDataset(train_paths), batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    test_dataloader = DataLoader(ImageDataset(test_paths), batch_size=int(batch_size), shuffle=True, num_workers=n_cpu)

    return train_dataloader, test_dataloader