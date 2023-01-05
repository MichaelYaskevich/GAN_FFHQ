import math
import torch
import torch.nn as nn

from scr.help_data_structures.exceptions import WrongImageSizeException


class GanModel:
    def __init__(self, image_size, latent_size):
        if image_size < 64 or int(math.log2(image_size)) != math.log2(image_size):
            raise WrongImageSizeException(
                'Wrong size of the image. ' +
                'For this architecture size should be a power of two and not less than 64.')
        self._image_size = image_size
        self._latent_size = latent_size
        self._device = torch.device('cpu')

        convs_to_add_count = image_size // 64 - 1
        self._discriminator = create_discriminator(convs_to_add_count)
        self._generator = create_generator(convs_to_add_count, latent_size)

    def get_latent_size(self):
        return self._latent_size

    def get_image_size(self):
        return self._image_size

    def get_discriminator(self):
        return self._discriminator

    def get_generator(self):
        return self._generator

    def to(self, device):
        self._device = device
        self._discriminator = self._discriminator.to(device)
        self._generator = self._generator.to(device)

    def get_device(self):
        return self._device


def create_discriminator(convs_to_add_count):
    discriminator = nn.Sequential(
        # in: 3 x image_size x image_size
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 x image_size/2 x image_size/2

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 x image_size/4 x image_size/4

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 256 x image_size/8 x image_size/8

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True)
        # out: 512 x image_size/16 x image_size/16
    )
    for i in range(convs_to_add_count):
        discriminator.add_module(
            str(12 + 3 * i),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False))
        discriminator.add_module(
            str(13 + 3 * i), nn.BatchNorm2d(512))
        discriminator.add_module(
            str(14 + 3 * i), nn.LeakyReLU(0.2, inplace=True))
    # out: 512 x 4 x 4

    discriminator.add_module(
        str(12 + 3 * convs_to_add_count),
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False))
    # out: 1 x 1 x 1

    discriminator.add_module(str(13 + 3 * convs_to_add_count), nn.Flatten())
    discriminator.add_module(str(14 + 3 * convs_to_add_count), nn.Sigmoid())

    return discriminator


def create_generator(convs_to_add_count, latent_size):
    generator = nn.Sequential(
        # in: latent_size x 1 x 1
        nn.ConvTranspose2d(latent_size, 512,
                           kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True)
        # out: 512 x 4 x 4
    )
    for i in range(convs_to_add_count):
        generator.add_module(str(1 + 3 * i), nn.ConvTranspose2d(
            512, 512, kernel_size=4, stride=2, padding=1, bias=False))
        generator.add_module(str(2 + 3 * i), nn.BatchNorm2d(512))
        generator.add_module(str(3 + 3 * i), nn.ReLU(True))
    # out: 512 x image_size/16 x image_size/16

    layers_count = 1 + 3 * convs_to_add_count
    generator.add_module(str(layers_count), nn.ConvTranspose2d(
        512, 256, kernel_size=4, stride=2, padding=1, bias=False))
    generator.add_module(str(layers_count + 1), nn.BatchNorm2d(256))
    generator.add_module(str(layers_count + 2), nn.ReLU(True))
    # out: 256 x image_size/8 x image_size/8

    generator.add_module(str(layers_count + 3), nn.ConvTranspose2d(
        256, 128, kernel_size=4, stride=2, padding=1, bias=False))
    generator.add_module(str(layers_count + 4), nn.BatchNorm2d(128))
    generator.add_module(str(layers_count + 5), nn.ReLU(True))
    # out: 128 x image_size/4 x image_size/4

    generator.add_module(str(layers_count + 6), nn.ConvTranspose2d(
        128, 64, kernel_size=4, stride=2, padding=1, bias=False))
    generator.add_module(str(layers_count + 7), nn.BatchNorm2d(64))
    generator.add_module(str(layers_count + 8), nn.ReLU(True))
    # out: 64 x image_size/2 x image_size/2

    generator.add_module(str(layers_count + 9), nn.ConvTranspose2d(
        64, 3, kernel_size=4, stride=2, padding=1, bias=False))
    generator.add_module(str(layers_count + 10), nn.Tanh())
    # out: 3 x image_size x image_size

    return generator
