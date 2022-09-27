# TMLC-Assignment-Bark-Texture-Images-Classification

## Dataset Description 
### Context
BarkVN-50 consists of 50 categories of bark texture images. Total number is 5,578 images with 303× 404 pixels. Image classification task can be performed on this dataset.

### Acknowledgements

Truong Hoang, Vinh (2020), “BarkVN-50”, Mendeley Data, V1, doi: 10.17632/gbt4tdmttn.1

### Dataset Link 

 [Dataset](https://www.kaggle.com/datasets/saurabhshahane/barkvn50)
 
### Steps Performed For The Project
 
 ![image](https://user-images.githubusercontent.com/70902291/192508544-eb39e3c0-17d2-4822-a5cb-fd80b5e41879.png)
 
 ### Data Visualizations 
 
 
## Barplot of Number of Images of Each Species

![image](https://user-images.githubusercontent.com/70902291/192509361-b268e0f0-3e6d-4abd-b9a3-cc347e3f63f8.png)

- We can Clearly see That we have Total 50 Classes

## Sample Image From dataset 

![image](https://user-images.githubusercontent.com/70902291/192509622-0e8fb026-1161-474b-b513-110d2cc2b3b1.png)

### Image Datagenerator 

![image](https://user-images.githubusercontent.com/70902291/192509885-0a6b2689-8ba1-4eb1-b86e-adfd1780522e.png)

### Augmented Image 

![image](https://user-images.githubusercontent.com/70902291/192510031-9633c887-611e-4776-8655-fae8ef7eaa87.png)

## Transfer Learning 

### Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.

- Vgg16  
- Xception
- Resnet
- Inception V3

### Resnets Architect 

What is ResNet?
ResNet, short for Residual Network is a specific type of neural network that was introduced in 2015 by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun in their paper “Deep Residual Learning for Image Recognition”.The ResNet models were extremely successful which you can guess from the following:

- Won 1st place in the ILSVRC 2015 classification competition with a top-5 error rate of 3.57% (An ensemble model)
- Won the 1st place in ILSVRC and COCO 2015 competition in ImageNet Detection, ImageNet localization, Coco detection and Coco segmentation.
- Replacing VGG-16 layers in Faster R-CNN with ResNet-101. They observed relative improvements of 28%
- Efficiently trained networks with 100 layers and 1000 layers also.

## Need for ResNet
Mostly in order to solve a complex problem, we stack some additional layers in the Deep Neural Networks which results in improved accuracy and performance. The intuition behind adding more layers is that these layers progressively learn more complex features. For example, in case of recognising images, the first layer may learn to detect edges, the second layer may learn to identify textures and similarly the third layer can learn to detect objects and so on. But it has been found that there is a maximum threshold for depth with the traditional Convolutional neural network model. Here is a plot that describes error% on training and testing data for a 20 layer Network and 56 layers Network.

We can see that error% for 56-layer is more than a 20-layer network in both cases of training data as well as testing data. This suggests that with adding more layers on top of a network, its performance degrades. This could be blamed on the optimization function, initialization of the network and more importantly vanishing gradient problem. You might be thinking that it could be a result of overfitting too, but here the error% of the 56-layer network is worst on both training as well as testing data which does not happen when the model is overfitting.

![image](https://user-images.githubusercontent.com/70902291/192511205-0eac2483-9922-48a3-9710-7d09e4b8d3e2.png)


Residual Block
This problem of training very deep networks has been alleviated with the introduction of ResNet or residual networks and these Resnets are made up from Residual Blocks.


![image](https://user-images.githubusercontent.com/70902291/192511396-e59f9f58-c2b5-4eb4-b0b5-5b7cfed6a0f8.png)

### For More Visualization Visit the below site 
[Visualize Here](https://tensorspace.org/html/playground/resnet50.html)

- We have Trained Model using Resnet Architect and save it as res.hdf5.

![image](https://user-images.githubusercontent.com/70902291/192512034-5373f87c-7359-4245-a499-937aa772d303.png)

##  Testing Model on Own Image

![image](https://user-images.githubusercontent.com/70902291/192512390-8772aefb-1008-4864-8ca8-c8b1a466db72.png)







               
              
