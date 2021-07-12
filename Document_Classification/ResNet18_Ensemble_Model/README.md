## Usage
1. Run the create_folders() functions to create all the required folders in the current directory
2. To Train -> 
    1. Place the dataset to use for training in the 'images_dataset' folder.
    2. The format in which the dataset should be present in the folder is very specific.
        images_dataset/class_name/image_name 

        'class_name' should be one of the classes in the 'doc2label' dict in the constants.py file else there will be errors all around 
    3. Now use commands -> 'train_header', 'train_footer', 'train_right_half', 'train_left_half', 'train_holistic' to train 5 different models to each specific regions.
    Their weights are saved in the respective region/section wise folders in the 'saved_state_dict' folder.
    4. After you've trained the above five region specific models check if there weights are saved correctly in the respective folders.
    5. Then run 'train_ensemble' to train the ensemble model. It saves it's weights in the 'saved_state_dict/ensemble'
3. To Test -> 
    1. Place the images to test(to get predictions on) in the 'custom_images' folder.
    2. Run 'test' from command line to get the predictions on those images. The predicted labels along with the probabilites are saved in the 'custom_images_predicted' folder.