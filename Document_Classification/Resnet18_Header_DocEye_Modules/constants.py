import os
# number of epochs of training
n_epochs = 50
# name of the dataset
dataset_path = str(os.path.join(os.getcwd(), 'custom_images'))
# size of the batches
batch_size = 16
# adam: learning rate
lr = 0.00008
# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of second order momentum of gradient
b2 = 0.999
# epoch from which to start lr decay
decay_epoch = 100
# number of cpu threads to use during batch generation
n_cpu = 8
# high res. image height
hr_height = 64
# high res. image width
hr_width = 64
# number of image channels
channels = 1

path_to_save_weights = '/content/drive/MyDrive/resnet18_header_classification_with_pan_and_adhar'
hr_shape = (hr_height, hr_width)

doc2label = {
    'Payslips':0,
    'Invoices Generic':1,
    'Bank Statements':2,
    'Death Certificate':3,
    'Doctor_Prescription_Sample':4,
    'Form16':5,
    'Handwritten Forms':6,
    'Handwritten Tranining texts':7,
    'ITR':8,
    'Mall Customer Invoices Image sample':9,
    'Medical Record_sample':10,
    'Motor Claims_Garage Invoices sample':11,
    'PAN':12,
    'Aadhar':13,
    'Policy Schedule':14,
    'Tele MER':15
}
count = {
    'Payslips':0,
    'Invoices Generic':0,
    'Bank Statements':0,
    'Death Certificate':0,
    'Doctor_Prescription_Sample':0,
    'Form16':0,
    'Handwritten Forms':0,
    'Handwritten Tranining texts':0,
    'ITR':0,
    'Mall Customer Invoices Image sample':0,
    'Medical Record_sample':0,
    'Motor Claims_Garage Invoices sample':0,
    'PAN':0,
    'Aadhar':0,
    'Policy Schedule':0,
    'Tele MER':0
}