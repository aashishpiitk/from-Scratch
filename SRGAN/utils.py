from PIL import Image, ImageOps
import re
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch, torchvision
import math
import os

def resize_snip_super_resolve_patch(path_to_image, generator, destination_folder_path = os.path.join(os.getcwd(), 'super_resolved'), SR_factor=2):
    """
    Breaks the images into patches of 32x32 pixels, 
        super-resolves each patch, and then joins the super-resolved 
        patches into one image which is super-resolved
        
        Parameters:
            path_to_path (str) : path to image to be super-resolved
            destination_folder_path (str) : path to the folder where to store the super resolved images
            SR_factor (int) : factor by which we have to super resolve the images i.e 2x, 4x, 8x, 16x
            generator (initialized generator model) : generator is the model into which the image is fed and it super-resolves the image
    """
    
    num = re.compile('\w{1,}')
    path_to_image = str(path_to_image)

    img_path = path_to_image   
    img_orig = ImageOps.grayscale(Image.open(img_path))
    trans1 = transforms.ToTensor()
    imt_orig= trans1(img_orig)

    (_, m, n) = imt_orig.shape

    ## resizing image so that both the dimensions are multiple of 32
    f = 32
    trans4 = transforms.Resize((max(32,(m//f)*f),max(32,(n//f)*f)))
    imt = trans4(imt_orig)

    (_, m, n) = imt.shape

    ## snipping and super-resolving

    image_index = []
    SR_image_index = []
    for i in range(m//f):
        for j in range(n//f):
            image_patch = imt[0][i*32:(i+1)*(32), j*(32):(j+1)*(32)]
            image_index.append(image_patch)
            SR_patch = generator(image_patch.unsqueeze(0).unsqueeze(0))
            SR_image_index.append(SR_patch.squeeze(0).squeeze(0).detach().cpu())

    ## patching the super-resolved parts

    SR_m = m*SR_factor
    SR_n = n*SR_factor
    
    c=0
    SR_final_image = torch.zeros((SR_m,SR_n))
    
    for i in range(m//f):
        for j in range(n//f):
            SR_final_image[i*(64):(i+1)*(64), j*(64):(j+1)*(64)] = SR_image_index[c]
            c+=1


    index = num.findall(path_to_image.split('/')[-1])[0]
    ## saving both the super-resolved and original images in one folder
    save_image(SR_final_image, os.path.join(destination_folder_path, f"_SR_{index}.png"))
    save_image(imt, os.path.join(destination_folder_path, f"_LR_{index}.png"))
    

def save_state_dict(generator, discriminator, optimizer_G, optimizer_D, epoch, save_state_dict_path = os.path.join(os.getcwd(), 'saved_state_dicts')):
    '''Save the state dictionaries of the generator, discriminator, and optimizers'''
    torch.save({
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'epoch': epoch+1,
                }, f"{save_state_dict_path}/checkpt{epoch+1}.pt")

def load_state_dict(generator, discriminator, optimizer_G, optimizer_D, checkpoint):
    '''Loads the saved state dicts for the models and the optimizer'''
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

def create_folder(folder_name):
    '''Creates a new folder in the current directory named as folder_name
    
        Parameters:
            folder_name (string) : name of the folder to create
    '''
    folderpath = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        if os.path.exists(folderpath):
            print('Folder Created', folderpath)
    
def create_required_folders():
    '''Creates the default folders in the current directory'''

    create_folder('images_dataset')
    create_folder('low_resolution')
    create_folder('super_resolved')
    create_folder('saved_state_dicts')
