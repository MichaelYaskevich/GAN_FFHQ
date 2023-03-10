from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch

from scr.data_preparation.methods import denorm


def save_losses_plot(losses_discriminator: List[float], losses_generator: List[float],
                     figsize: (int, int), dir_to_save) -> None:
    plt.figure(figsize=figsize)
    plt.plot(losses_discriminator, '-')
    plt.plot(losses_generator, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.savefig(Path(dir_to_save, 'losses.png'))


def save_scores_plot(real_scores: List[float], fake_scores: List[float],
                     figsize: (int, int), dir_to_save) -> None:
    plt.figure(figsize=figsize)
    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real', 'Fake'])
    plt.title('Scores')
    plt.savefig(Path(dir_to_save, 'scores.png'))


def save_images_plot(images: torch.Tensor, stats: tuple, dir_to_save) -> None:
    """
    :param images: images to show on plot
    :param stats: variance and mean for normalization
    :param dir_to_save: path to save png
    """
    nrows = images.size(0) // 2 + images.size(0) % 2
    fig, ax = plt.subplots(nrows, 2, figsize=(8, 8), constrained_layout=True)
    for i, image in enumerate(denorm(images, stats).cpu().detach()):
        ax[i // 2, i % 2].imshow(image)
    plt.savefig(Path(dir_to_save, 'few_generated_images.png'))


def save_tsne_results(generated, real, figsize, dir_to_save) -> None:
    plt.figure(figsize=figsize)
    plt.scatter(generated[:, 0], generated[:, 1], label='generated')
    plt.scatter(real[:, 0], real[:, 1], label='real')
    plt.legend()
    plt.savefig(Path(dir_to_save, 'tsne_results.png'))
