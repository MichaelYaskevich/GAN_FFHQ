import os
import shutil
import torch


from pathlib import Path
from scr.gan_architecture.GanModel import GanModel


def save_weights(gan: GanModel, path_to_save):
    """ Saves weights of the GanModel"""

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    torch.save(gan.get_generator().state_dict(), Path(path_to_save, 'generator.pt'))
    torch.save(gan.get_discriminator().state_dict(), Path(path_to_save, 'discriminator.pt'))
    shutil.make_archive(f'GanModel{gan.get_image_size()}x{gan.get_image_size()}', 'zip', path_to_save)
