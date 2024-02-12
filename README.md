# Image Classification by Deep Learning
The model classifies image as strawberry, tomato, and cherry.

This project demonstrates the transfer learning by fine-tuning a pre-trained model of image classification for the downstream task of strawberry, tomato, and cherry classification.
Pretrained model resnet18 is available under module torchvision.models in python. It is one of deep residual learning framework based on article by He et al. in 2015 [1]. Network is pretrained on ImageNet dataset. There are 18 deep layers, with the final layer configured to fine tune for the downstream task. 

### Fine-tune detail

*DATA*

There are around 4500 images of strawberry, tomato, and cherry. Particularly, there are 1,480 images of each tomato and cherry, and 1479 images of strawberry. Dataset is for education and not allowed to shared publicly.

*MODEL*

There are two options were experimented: 

* Freeze option: freeze parameters of previous layers, only update the last fully connected layer
* Unfreeze option: update the parameters of all layers.

In terms of time elapse, the freeze option takes 10 minutes to run the 1st epoch, the subsequent epochs take around 20 seconds each. The unfreeze option takes around 30 seconds each epoch. 

Interm of accuracy, after 10 epochs, the unfreeze option returns the higher accuracy of 94% and 91% for training and validating respectively, compared to 87% and 88% for training and validating respectively of the freeze option.

### Training

The training set is split into train and validate set by ratio 3:1. Validating set is run in each training epoch to measure the performance of dataset to unseen data and spot the potential overfit issue. The task is multi-class classification, the final layer of model is using softmax activation function which returns the probability for each class of the image. The predicted class will be the one with the highest probability. The training loss is calculated by Cross-Entropy (also known as log loss). This loss measures the divergence of predicted probability from the actual class. The main evaluation for classification is accuracy metric.

The pretrained Resnet18 (unfree option) is trained with 30 epochs to achieve the optimization result, there is no sign of significant overfitting. (Resnet18 with the skip connection design has assist the model in deeper 
design without prone to vanishing gradient, shared weightings by parallel learning feature maps help avoiding the overfitting issue.)

![image](https://github.com/tramchau/Image-Classification-by-DL/assets/17041836/d854a7bc-6363-46bb-99d5-bee3a17a8036)

Hyperparameters:

* batch size = 32, 
* optimize = SGD, 
* learning rate = 0.001, 
* weight decay = 0.01

### Performance
98% and 96% accuracy for training and 
validating respectively without significant overfitting.
### Inference

```{python}
!python inference.py
```

![image](https://github.com/tramchau/Image-Classification-by-DL/assets/17041836/ad7fa614-b485-4209-aeab-e4a708013724)

###  Copyright of inference images used for illustration:

From left to right:\
1  Copyright: Freeimages.com / hgec\
2  Copyright: Freeimages.com / alainap\
3  Copyright: Freeimages.com / mm904ut\
4  Copyright: Freeimages.com / DartVader\
5  Copyright: Freeimages.com / shibuya86

### Reference
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” 2015
