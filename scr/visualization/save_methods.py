import matplotlib.pyplot as plt

from scr.data_preparation.methods import denorm


def save_losses_plot(losses_discriminator, losses_generator, figsize, path_to_save):
    plt.figure(figsize=figsize)
    plt.plot(losses_discriminator, '-')
    plt.plot(losses_generator, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.savefig(path_to_save)


def save_scores_plot(real_scores, fake_scores, figsize, path_to_save):
    plt.figure(figsize=figsize)
    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real', 'Fake'])
    plt.title('Scores')
    plt.savefig(path_to_save)


def save_images_plot(generated, stats, path_to_save):
    nrows = generated.size(0) // 2 + generated.size(0) % 2
    fig, ax = plt.subplots(nrows, 2, figsize=(8, 8), constrained_layout=True)
    for i, image in enumerate(denorm(generated, stats).cpu().detach()):
        ax[i // 2, i % 2].imshow(image)
    plt.savefig(path_to_save)


def save_tsne_results(generated, real, figsize, path_to_save):
    plt.figure(figsize=figsize)
    plt.scatter(generated[:, 0], generated[:, 1], label='generated')
    plt.scatter(real[:, 0], real[:, 1], label='real')
    plt.legend()
    plt.savefig(path_to_save)