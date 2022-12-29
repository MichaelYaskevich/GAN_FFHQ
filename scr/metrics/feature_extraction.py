import torch.nn as nn


def get_feature_extractor():
    return nn.Sequential(
        # in: 3 x image_size x image_size
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 x image_size/2 x image_size/2

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 x image_size/4 x image_size/4

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        # out: 256 x image_size/8 x image_size/8
    )
