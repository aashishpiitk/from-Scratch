import os
# number of epochs of training
n_epochs = 50
# paths to images for testing and training
dataset_path_to_train = os.path.join(os.getcwd(), 'images_dataset')
folder_path_to_save_predicted_images = os.path.join(os.getcwd(), 'custom_images_predicted')
folder_path_to_test_images = os.path.join(os.getcwd(), 'custom_images')
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

path_to_save_weights = {
    'header' : os.path.join(os.getcwd(), 'saved_state_dicts', 'header'),
    'footer' : os.path.join(os.getcwd(), 'saved_state_dicts', 'footer'),
    'right_half' : os.path.join(os.getcwd(), 'saved_state_dicts', 'right_half'),
    'left_half' : os.path.join(os.getcwd(), 'saved_state_dicts', 'left_half'),
    'holistic' : os.path.join(os.getcwd(), 'saved_state_dicts', 'holistic'),
    'ensemble' : os.path.join(os.getcwd(), 'saved_state_dict', 'ensemble')
}

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
label2doc = {
    0:'Payslips',
    1:'Invoices Generic',
    2:'Bank Statements',
    3:'Death Certificate',
    4:'Doctor_Prescription_Sample',
    5:'Form16',
    6:'Handwritten Forms',
    7:'Handwritten Tranining texts',
    8:'ITR',
    9:'Mall Customer Invoices Image sample',
    10:'Medical Record_sample',
    11:'Motor Claims_Garage Invoices sample',
    12:'PAN',
    13:'Aadhar',
    14:'Policy Schedule',
    15:'Tele MER'
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