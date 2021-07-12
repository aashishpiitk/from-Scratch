# Document Classfication Using Transfer Learning
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
    
## Help Log
## References

1. [Sample Notebook](https://github.com/sambalshikhar/Document-Image-Classification-with-Intra-Domain-Transfer-Learning-and-Stacked-Generalization-of-Deep/blob/master/document_classification_resub2.ipynb)
2. [Repository of the Author of the Paper](https://github.com/hiarindam/document-image-classification-TL-SG/issues/4)
3. [Main Paper Used](https://arxiv.org/abs/1801.09321)
4. [Parent Paper of the Main Paper Used](https://arxiv.org/pdf/1502.07058.pdf)

## Experimentation
Exprimented with ResNet18 and VGG16 models on RVL-CDIP test dataset.
First Trained the classifier on the whole images and then compared the accuracy.
VGG116 holistic-> 82 percent
ResNet18 holistic-> 85-86 percent
ResNet18 ensemble -> 92-93 percent

Then choose ResNet18 to train it on the four differnt regions of the images-
Header
Footer
Right Half
Left Half

Upon Ensemble of these ResNet18 attained an accuracy of 92-93 percent on RVL-CDIP dataset.

Then moved to testing and fine tuning the model on the EigenLytics dataset.

|Experiment                                     |Accuracy                                  |
|---------------                                |:-------------:                           |
|ResNet18 Holistic on RVL-CDIP                  |85%                                       |
|ResNet18 Header on RVL-CDIP                    |31%                                       |
|ResNet18 Footer on RVL-CDIP                    |70%                                       |
|ResNet18 Right Half on RVL-CDIP                |75%                                       |
|ResNet18 Left Half on RVL-CDIP                 |77%                                       |
|ResNet18 Ensemble on RVL-CDIP                  |92-93%                                    |
 


## Critical Observations
### Why ResNet18 over VGG16?
1. [ResNet18 has faster training speed than VGG16.](https://stats.stackexchange.com/questions/280179/why-is-resnet-faster-than-vgg/280338#)
2. Better Accuracy of ResNet18(slightly better, may not be considered as a deciding factor)

## Idea Implementation Details
1. Train/Fine-Tune(all layers) ResNet18 to classify images based on specific regions --> header, footer, right_half, left_half, holistic(full page)
2. So in total one has to fine-tune 5 ResNet18 models and save its weights in a folder.
3. Then for the final classfication(ensemble model) we load all the models including the new meta_classifier in memory, and load the weights for all except the meta_classifier(as it is not trained yet).
    The predictions for each section are tensors of size 16(number of document classes) they are cocatenated into a single tensor(5*16), which is finally fed into the meta_classifier.
    Meta_classifier is just a name given to a neural network/any other classifier which can be used as a final classification layer in the ensemble model.
4. Then train the meta_classifier. Make sure that gradients are not propagated to the all the other models which are loaded into the memory(use detach() for this).
