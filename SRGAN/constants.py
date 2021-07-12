import os

# number of epochs of training
n_epochs = 50
# size of the batches
batch_size = 64
# adam: learning rate
lr = 0.00008
# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of second order momentum of gradient
b2 = 0.999
# epoch from which to start lr decay
decay_epoch = 100
# number of cpu threads to use during batch generation
n_cpu = 2
# high res. image height
hr_height = 64
# high res. image width
hr_width = 64
# number of image channels
channels = 1
# Shape of cropped patch of image which is to be treated as HR image during training
hr_shape = (hr_height, hr_width)
# path of the dataset
dataset_path = str(os.path.join(os.getcwd(), 'images_dataset'))
# path to save the state dict
save_state_dict_path = str(os.path.join(os.getcwd(), 'saved_state_dicts'))
# path to the folder where low resolution images are stored and which need to be super-resolved
low_resolution_images_path = str(os.path.join(os.getcwd(), 'low_resolution'))
# path to the folder of super-resolved images
super_resolved_images_path = str(os.path.join(os.getcwd(), 'super_resolved'))
