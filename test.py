import torch
from torchvision import datasets, transforms

import argparse
import os
from tqdm import tqdm

from model import MobileNetV3
from preprocess import load_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def main():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--batch-size', type=int, default=16, help='batch size, (default: 100)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR100", help="which dataset you use, (example: CIFAR10, CIFAR100), (default: CIFAR100)")
    parser.add_argument('--is-train', type=bool, default=False, help="True if training, False if test. (default: False)")
    parser.add_argument('--model-mode', type=str, default="LARGE", help="(example: LARGE, SMALL), (default: LARGE)")

    args = parser.parse_args()

    _, test_loader = load_data(args)

    if args.dataset_mode == "CIFAR100":
        num_classes = 100
    elif args.dataset_mode == "CIFAR10":
        num_classes = 10

    if os.path.exists("./checkpoint"):
        model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes).to(device)
        filename = "best_model_"
        checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        end_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        print("[Saved Best Accuracy]: ", best_acc, '%', "[End epochs]: ", end_epoch)

        model.eval()
        correct = 0
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(target.data).sum()
        print("[Test Accuracy] ", 100. * float(correct) / len(test_loader.dataset), '%')

    else:
        assert os.path.exists("./checkpoint/" + str(args.seed) + "ckpt.t7"), "File not found. Please check again."
    print("Number of model parameters: ", get_model_parameters(model))


if __name__ == "__main__":
    main()
