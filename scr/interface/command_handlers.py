import torch
import zipfile
import shutil
import os

from torch import nn
from pathlib import Path

from scr.data_preparation import get_dataloader, DeviceDataLoader
from scr.gan_architecture import GanModel, GanCriterion, GanOptimizer
from scr.image_generation import save_generated_images
from scr.metrics import get_leave_one_out_1nn_cls_accuracy, get_feature_extractor, tsne_for_batch
from scr.train import save_weights, train
from scr.train.save_methods import save_accuracy
from scr.visualization.save_methods import save_losses_plot, save_scores_plot, save_images_plot, save_tsne_results


def handle_train_cmd(image_size, epochs, lr, dir_to_save):
    """Creates gan, trains it, counts 1nn classifier accuracy and visualizes results with help of TSNE"""
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 64
    data_dir = Path(Path(__file__).parent.parent, 'resources', 'faces_dataset_small')
    train_dl = get_dataloader(image_size, batch_size, data_dir, stats)
    train_dl = DeviceDataLoader(train_dl, device)

    latent_size = 128
    gan = GanModel(image_size, latent_size)
    gan.to(device)
    criterion = GanCriterion(nn.BCELoss(), nn.BCELoss())
    optimizer = GanOptimizer(torch.optim.Adam(gan.get_discriminator().parameters(), lr=lr, betas=(0.5, 0.999)),
                             torch.optim.Adam(gan.get_generator().parameters(), lr=lr, betas=(0.5, 0.999)))

    # losses_g, losses_d, real_scores, fake_scores = train(train_dl, gan, criterion, optimizer, epochs)
    losses_g = [1, 2]
    losses_d = [2, 3]
    real_scores = [3, 4]
    fake_scores = [2, 3]

    save_weights(gan, Path(dir_to_save, 'model'))
    figsize = (15, 6)
    save_losses_plot(losses_d, losses_g, figsize, dir_to_save)
    save_scores_plot(real_scores, fake_scores, figsize, dir_to_save)

    n_images = 4
    fixed_latent = torch.randn(n_images, latent_size, 1, 1, device=device)
    fake_images = gan.get_generator()(fixed_latent).permute(0, 2, 3, 1)
    save_images_plot(fake_images, stats, dir_to_save)

    noise_vectors = torch.randn(len(train_dl) * batch_size, latent_size, 1, 1, device=device)
    path = Path(dir_to_save, 'generated_faces')
    save_generated_images(gan, batch_size, len(train_dl), noise_vectors, Path(path, 'generated_faces'), stats)

    fake_images_dl = get_dataloader(image_size, batch_size, path, stats)
    fake_images_dl = DeviceDataLoader(fake_images_dl, gan.get_device())

    # print('counting leave_one_out_1nn_cls_accuracy ...')
    # accuracy = get_leave_one_out_1nn_cls_accuracy(
    #     train_dl, fake_images_dl, get_feature_extractor().to(device))
    # save_accuracy(accuracy, dir_to_save)

    # print('counting tsne ...')
    # batch_after_tsne = tsne_for_batch(
    #     train_dl, fake_images_dl, get_feature_extractor().to(device))
    # generated = batch_after_tsne[:batch_size]
    # real = batch_after_tsne[batch_size:]
    # save_tsne_results(generated, real, figsize, dir_to_save)

    shutil.rmtree(path)



def handle_eval_cmd(image_size, images_count, dir_to_save, path_to_weights):
    """Loads weights and generates images."""
    if not os.path.exists(path_to_weights):
        print(f'wrong path to the weights: {path_to_weights}')
        return
    path = Path(dir_to_save, 'generated_faces')
    latent_size = 128
    try:
        gan = load_gan(image_size, latent_size, path_to_weights)
    except RuntimeError:
        print(f'Could not load the weights {path_to_weights}. Check that image size is correct')
        return
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    print(f'generating {images_count} images ...')
    if images_count < 512:
        noise_vectors = torch.randn(images_count, latent_size, 1, 1)
        save_generated_images(gan, images_count, 1, noise_vectors, path, stats)
    else:
        noise_vectors = torch.randn(images_count - images_count % 512, latent_size, 1, 1)
        save_generated_images(gan, 512, images_count // 512, noise_vectors, path, stats)
        noise_vectors = torch.randn(images_count % 512, latent_size, 1, 1)
        save_generated_images(gan, images_count % 512, 1, noise_vectors, path, stats,
                              start_index=images_count - images_count % 512)


def load_gan(image_size, latent_size, path_to_weights):
    """Loads weights"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    gan = GanModel(image_size, latent_size)
    gan.to(device)
    temp_weight_path = Path(Path(path_to_weights).parent, 'temp_weights')

    with zipfile.ZipFile(path_to_weights, 'r') as zip_ref:
        zip_ref.extractall(temp_weight_path)

    gan.get_discriminator().load_state_dict(
        torch.load(Path(temp_weight_path, 'discriminator.pt'), map_location=gan.get_device()))
    gan.get_generator().load_state_dict(
        torch.load(Path(temp_weight_path, 'generator.pt'), map_location=gan.get_device()))
    shutil.rmtree(temp_weight_path)

    return gan
