#Introduction

#Index
1. How to use
2. Purpose
3. Structure of file
4. What is Resnet
5. Reference



#Contents
1. How to use
->Run Evaluation.py after build all file

2. Purpose
->For checking effectivity of Resnet

3. Structure of file
- Evaluation.py is for evaluate resnet effevyivity
- Resnet.py is for combining resnet base module
- Resnet_basic_module is for Resnet's basic component which include batch normalization, convolution layer, and Relu.


4. What is Resnet
Resnet is used for image processing and the one of the Deep learning model which make it possible to take more deep structure[1].
Before Resnet is arrival, it is difficult to take deep structure because of Vanishing Gradient.
However Resnet mitigate this Vanishing Gradient problem by using residual function.
We can shorten the network by using residual function, it will stop significant decrease of Gradient in deep neural network.

5. Reference
[1]K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 770-778.
doi: 10.1109/CVPR.2016.90
