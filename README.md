# Image_Colorization
A deep learning model which colors a black and white image.

Dataset - Any rgb image can be used a valid dataset in this project. In order for the model to be robust, I first used http://www.vision.caltech.edu/Image_Datasets/Caltech256/ dataset for making the backbone architecture VGG16_bn learn gray scale image clssification.After learning Vgg16_bn for gray scale images, U-Net was trained for semantic segmentation task. Then different datasets were used to train for colorization. Model was hoped to learn places,faces,objects etc.

Language Used - Python

Framework Used - Pytorch

Architecture - Pretrained U-Net on gray scale images and then segmentation , was used for colorization seen as a regression problem. On going work tries to improve the results by using classification as the fundamental task for image colorization. The image can be divided into discrete bins of color ranges once converted to LAB color space. A total of 313 discrete bins are genrated by dividing the probabilty distribution into 10x10 sized bins. Then a classification task is tried to be solved. 
