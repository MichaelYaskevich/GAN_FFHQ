import os
import shutil
import torch

from pathlib import Path

from scr.gan_architecture.GanModel import GanModel


def save_weights(gan: GanModel, path_to_save) -> None:
    """ Saves weights of the GanModel"""

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    torch.save(gan.get_generator().state_dict(), Path(path_to_save, 'generator.pt'))
    torch.save(gan.get_discriminator().state_dict(), Path(path_to_save, 'discriminator.pt'))
    name = f'GanModel{gan.get_image_size()}x{gan.get_image_size()}'
    shutil.make_archive(str(Path(Path(path_to_save).parent, name)), 'zip', path_to_save)


def save_accuracy(accuracy, path_to_save):
    with open(Path(path_to_save, 'accuracy.txt'), 'w') as f:
        f.write(str(accuracy))
