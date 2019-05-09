# Implementing Searching for MobileNetV3 paper using Pytorch
- The current model is a very early model. I will modify it as a general model as soon as possible.
## Paper
- [Searching for MobileNetV3 paper](https://arxiv.org/abs/1905.02244)
- Author: Andrew Howard(Google Research), Mark Sandler(Google Research, Grace Chu(Google Research), Liang-Chieh Chen(Google Research), Bo Chen(Google Research), Mingxing Tan(Google Brain), Weijun Wang(Google Research), Yukun Zhu(Google Research), Ruoming Pang(Google Brain), Vijay Vasudevan(Google Brain), Quoc V. Le(Google Brain), Hartwig Adam(Google Research)

## Todo
- Experimental need for ImageNet dataset.
- Code refactoring

## MobileNetV3 Block
![캡처](https://user-images.githubusercontent.com/22078438/57360577-6f30d000-71b5-11e9-89a6-24034a3ecdde.PNG)

## Experiments
- For CIFAR-100 data, I experimented with resize (224, 224).<br>

| Datasets | Model | Accuracy | Epoch | Training Time | Parameters
| :---: | :---: | :---: | :---: | :---: | :---: |
CIFAR-100 | MobileNetV3(LARGE) | 68.99% | 63 | 3h 39m | 3.99M
CIFAR-100 | MobileNetV3(SMALL) | | | |
IMAGENET | MobileNetV3(LARGE) WORK IN PROCESS | | | | 5.15M
IMAGENET | MobileNetV3(SMALL) WORK IN PROCESS | | | | 2.94M

## Requirements
- torch==1.0.1
