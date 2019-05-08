import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class bneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, HS, RE, SE, dense_layer, padding=1):
        super(bneck, self).__init__()
        self.out_channels = out_channels
        self.HS = HS
        self.RE = RE
        self.SE = SE

        self.use_connect = stride == 1 and in_channels == out_channels

        if self.RE == True:
            self.activation_RE = nn.ReLU(inplace=True)
        elif self.HS == True:
            self.activation_HS = nn.ReLU6(inplace=True)

        if self.SE == True:
            self.squeeze = nn.Sequential(
                nn.Linear(in_channels, dense_layer),
                nn.ReLU(inplace=True),
                nn.Linear(dense_layer, in_channels),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernal_size, stride=stride, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # MobileNetV2
        output = self.conv(x)
        if self.RE == True:
            output = self.activation_RE(output)
        elif self.HS == True:
            output = self.activation_HS(output + 3) / 6 * output
        batch, channels, height, width = output.size()
        original_output = output
        if self.SE == True:
            # Squeeze and Excite
            global_pooled_output = F.avg_pool2d(output, kernel_size=[height, width])
            global_pooled_output = torch.reshape(global_pooled_output, shape=(-1, channels))
            squeeze = self.squeeze(global_pooled_output)
            squeeze = torch.reshape(squeeze, shape=(-1, channels, 1, 1))
            squeeze = torch.sigmoid(squeeze) * squeeze
            output = original_output * squeeze

        output = self.conv1x1(output)
        if self.RE == True:
            output = self.activation_RE(output)
        elif self.HS == True:
            output = self.activation_HS(output + 3) / 6 * output
        if self.use_connect:
            return x + output
        else:
            return output


class MobileNetV3(nn.Module):
    def __init__(self, model_mode="LARGE", num_classes=1000):
        super(MobileNetV3, self).__init__()
        self.activation_HS = nn.ReLU6(inplace=True)
        self.num_classes = num_classes

        if model_mode == "LARGE":
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
            )

            self.block = nn.Sequential(
                bneck(in_channels=16, out_channels=16, kernal_size=3, stride=1, HS=False, RE=True, SE=False, dense_layer=16),
                bneck(in_channels=16, out_channels=24, kernal_size=3, stride=2, HS=False, RE=True, SE=False, dense_layer=64),
                bneck(in_channels=24, out_channels=24, kernal_size=3, stride=1, HS=False, RE=True, SE=False, dense_layer=72),
                bneck(in_channels=24, out_channels=40, kernal_size=5, stride=2, HS=False, RE=True, SE=True, dense_layer=72, padding=2),
                bneck(in_channels=40, out_channels=40, kernal_size=5, stride=1, HS=False, RE=True, SE=True, dense_layer=120, padding=2),
                bneck(in_channels=40, out_channels=40, kernal_size=5, stride=1, HS=False, RE=True, SE=True, dense_layer=120, padding=2),
                bneck(in_channels=40, out_channels=80, kernal_size=3, stride=2, HS=True, RE=False, SE=False, dense_layer=240),
                bneck(in_channels=80, out_channels=80, kernal_size=3, stride=1, HS=True, RE=False, SE=False, dense_layer=200),
                bneck(in_channels=80, out_channels=80, kernal_size=3, stride=1, HS=True, RE=False, SE=False, dense_layer=184),
                bneck(in_channels=80, out_channels=80, kernal_size=3, stride=1, HS=True, RE=False, SE=False, dense_layer=184),
                bneck(in_channels=80, out_channels=112, kernal_size=3, stride=1, HS=True, RE=False, SE=True, dense_layer=480),
                bneck(in_channels=112, out_channels=112, kernal_size=3, stride=1, HS=True, RE=False, SE=True, dense_layer=672),
                bneck(in_channels=112, out_channels=160, kernal_size=5, stride=1, HS=True, RE=False, SE=True, dense_layer=672, padding=2),
                bneck(in_channels=160, out_channels=160, kernal_size=5, stride=2, HS=True, RE=False, SE=True, dense_layer=672, padding=2),
                bneck(in_channels=160, out_channels=160, kernal_size=5, stride=1, HS=True, RE=False, SE=True, dense_layer=960, padding=2),
            )

            self.conv1 = nn.Sequential(
                nn.Conv2d(160, 960, kernel_size=1, stride=1),
                nn.BatchNorm2d(960),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(960, 1280, kernel_size=1, stride=1),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(1280, self.num_classes, kernel_size=1, stride=1),
            )

        elif model_mode == "SMALL":
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
            )

            self.block = nn.Sequential(
                bneck(in_channels=16, out_channels=16, kernal_size=3, stride=2, HS=False, RE=True, SE=True, dense_layer=16),
                bneck(in_channels=16, out_channels=24, kernal_size=3, stride=2, HS=False, RE=True, SE=False, dense_layer=72),
                bneck(in_channels=24, out_channels=24, kernal_size=3, stride=1, HS=False, RE=True, SE=False,dense_layer=88),
                bneck(in_channels=24, out_channels=40, kernal_size=5, stride=2, HS=True, RE=False, SE=True, dense_layer=96, padding=2),
                bneck(in_channels=40, out_channels=40, kernal_size=5, stride=1, HS=True, RE=False, SE=True, dense_layer=240, padding=2),
                bneck(in_channels=40, out_channels=40, kernal_size=5, stride=1, HS=True, RE=False, SE=True, dense_layer=240, padding=2),
                bneck(in_channels=40, out_channels=48, kernal_size=5, stride=1, HS=True, RE=False, SE=True, dense_layer=120, padding=2),
                bneck(in_channels=48, out_channels=48, kernal_size=5, stride=1, HS=True, RE=False, SE=True, dense_layer=144, padding=2),
                bneck(in_channels=48, out_channels=96, kernal_size=5, stride=2, HS=True, RE=False, SE=True, dense_layer=288, padding=2),
                bneck(in_channels=96, out_channels=96, kernal_size=5, stride=1, HS=True, RE=False, SE=True, dense_layer=576, padding=2),
                bneck(in_channels=96, out_channels=96, kernal_size=5, stride=1, HS=True, RE=False, SE=True, dense_layer=576, padding=2),
            )

            self.conv1 = nn.Sequential(
                nn.Conv2d(96, 576, kernel_size=1, stride=1),
                nn.BatchNorm2d(576),
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(576, 1280, kernel_size=1, stride=1),
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(1280, self.num_classes, kernel_size=1, stride=1),
            )

        self.apply(_weights_init)

    def forward(self, x):
        output = self.init_conv(x)
        output = self.activation_HS(output + 3) / 6 * output
        output = self.block(output)
        output = self.conv1(output)
        output = self.activation_HS(output + 3) / 6 * output
        batch, channels, height, width = output.size()
        output = F.avg_pool2d(output, kernel_size=[height, width])
        output = self.conv2(output)
        output = self.activation_HS(output + 3) / 6 * output
        output = self.conv3(output)
        output = torch.reshape(output, shape=(-1, self.num_classes))
        return output


# temp = torch.zeros((1, 3, 224, 224))
# model = MobileNetV3(model_mode="SMALL")
# print(model(temp).shape)
