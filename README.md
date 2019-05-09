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

## Usage

### Train
```
python main.py
```
- If you want to change hyper-parameters, you can check "python main.py --help"

Options:
- `--dataset-mode` (str) - which dataset you use, (example: CIFAR10, CIFAR100), (default: CIFAR100).
- `--epochs` (int) - number of epochs, (default: 100).
- `--batch-size` (int) - batch size, (default: 128).
- `--learning-rate` (float) - learning rate, (default: 1e-1).
- `--dropout` (float) - dropout rate, (default: 0.3).
- `--model-mode` (str) - which network you use, (example: LARGE, SMALL), (default: LARGE).
- `--load-pretrained` (bool) - (default: False).

### Test
```
python test.py
```
- Put the saved model file in the checkpoint folder and saved graph file in the saved_graph folder and type "python test.py".
- If you want to change hyper-parameters, you can check "python test.py --help"
- The model file currently in the checkpoint folder is a model with an accuracy of 92.70%.

Options:
- `--model-mode` (str) - which network you use, (example: LARGE, SMALL), (default: LARGE).
- `--batch-size` (int) - batch size, (default: 128).
- `--dataset-mode` (str) - which dataset you use, (example: CIFAR10, CIFAR100), (default: CIFAR100).
- `--is-train` (bool) - True if training, False if test. (default: False).

### Number of Parameters
```python
import torch

from model import MobileNetV3

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

tmp = torch.randn((128, 3, 224, 224))
model = MobileNetV3(model_mode="LARGE")
print("Number of model parameters: ", get_model_parameters(model))
```

## Requirements
- torch==1.0.1
