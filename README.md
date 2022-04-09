# An Overhead-Free Region-Based JPEG Framework for Task-Driven Image Compression

<p align='center'>
    <img src='./figures/fig1.png' width='600px'/>
</p>

## Abstract
>An increasing amount of captured images are streamed to a remote server or stored in a device for deep neural network (DNN) inference. In most cases, raw images are compressed with encoding algorithms such as JPEG to cope with resource limitations. However, the standard JPEG optimized for human visual systems may induce significant accuracy loss in DNN inference tasks. In addition, the standard JPEG compresses all regions in an image at the same quality level, while some areas may not contain valuable information for the target task. In this paper, we propose a target-driven JPEG compres- sion framework that performs region-adaptive quantization of the DCT coefficients. The region-based quality map is generated from an end-to-end trainable neural network. In addition, we present a deep learning approach to remove the requirement of storing the overhead information induced by the region-based encoding process. Our framework can be easily implemented on devices with commonly used JPEG and also produce images that achieve a higher compression rate with minimum degradation of the classification accuracy.

## Requirements
* TensorFlow 2.5 +


## Training on TPUs
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
    https://colab.research.google.com/github/jshye/qf-map-generator/blob/master/train.ipynb)
