# Implementing Searching for MobileNetV3 paper using Pytorch
- The current model is a very early model.  will modify it as a general model as soon as possible.
## Paper
- [Searching for MobileNetV3 paper](https://arxiv.org/abs/1905.02244)
- Author: Andrew Howard(Google Research), Mark Sandler(Google Research, Grace Chu(Google Research), Liang-Chieh Chen(Google Research), Bo Chen(Google Research), Mingxing Tan(Google Brain), Weijun Wang(Google Research), Yukun Zhu(Google Research), Ruoming Pang(Google Brain), Vijay Vasudevan(Google Brain), Quoc V. Le(Google Brain), Hartwig Adam(Google Research)

## Todo
- Experimental need for ImageNet dataset.
- Code refactoring

## MobileNetV3 Block
![캡처](https://user-images.githubusercontent.com/22078438/57360577-6f30d000-71b5-11e9-89a6-24034a3ecdde.PNG)

## Experiments
- For CIFAR-10 and CIFAR-100 data, I experimented with resize (224, 224).<br>

| Datasets | Model | Accuracy | Epoch | Training Time | Parameters
| :---: | :---: | :---: | :---: | :---: | :---: |
CIFAR-100 | MobileNetV3(LARGE) | 64.64% | 62 | 1h 56m | 2.5M
CIFAR-100 | MobileNetV3(SMALL) | 62.29% | 86 | 2h 17m | 1.2M
IMAGENET | WORK IN PROCESS | | | |

## Requirements
- torch==1.0.1
