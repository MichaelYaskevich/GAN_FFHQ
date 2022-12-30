import torch
import numpy as np

from tqdm.notebook import tqdm

from scr.data_preparation import DeviceDataLoader
from scr.gan_architecture import GanModel, GanOptimizer, GanCriterion


def get_images_loss_and_score(gan: GanModel, images: torch.Tensor,
                              criterion: GanCriterion, is_real: bool) -> (torch.Tensor, float):
    preds = gan.get_discriminator()(images)
    create_targets = torch.ones if is_real else torch.zeros
    targets = create_targets(images.size(0), 1, device=gan.get_device())
    loss = criterion.get_discriminator_loss()(preds, targets)
    score = torch.mean(preds).item()
    return loss, score


def train_discriminator(gan: GanModel, optimizer: GanOptimizer, criterion: GanCriterion,
                        real_images: torch.Tensor) -> (float, float, float):
    """Counts discriminator loss and makes optimizer step"""
    device = gan.get_device()
    optimizer.get_discriminator_optimizer().zero_grad()

    real_loss, cur_real_score = get_images_loss_and_score(
        gan, real_images, criterion, is_real=True)

    latent = torch.randn(real_images.size(0), gan.get_latent_size(), 1, 1, device=device)
    fake_images = gan.get_generator()(latent)

    fake_loss, cur_fake_score = get_images_loss_and_score(
        gan, fake_images, criterion, is_real=False)

    loss_d = real_loss + fake_loss
    loss_d.backward()
    optimizer.get_discriminator_optimizer().step()

    return cur_real_score, cur_fake_score, loss_d.item()


def train_generator(gan: GanModel, optimizer: GanOptimizer,
                    criterion: GanCriterion, batch_size: int) -> float:
    """Tries to fool the discriminator with fake images, counts loss and makes optimizer step"""
    device = gan.get_device()
    optimizer.get_generator_optimizer().zero_grad()

    latent = torch.randn(batch_size, gan.get_latent_size(), 1, 1, device=device)
    fake_images = gan.get_generator()(latent)

    loss_g, _ = get_images_loss_and_score(
        gan, fake_images, criterion, is_real=True)

    loss_g.backward()
    optimizer.get_generator_optimizer().step()

    return loss_g.item()


def train(train_dl: DeviceDataLoader, gan: GanModel, criterion: GanCriterion,
          optimizer: GanOptimizer, epochs: int):
    gan.get_discriminator().train()
    gan.get_generator().train()
    torch.cuda.empty_cache()

    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    for epoch in range(epochs):
        loss_d_per_epoch = []
        loss_g_per_epoch = []
        real_score_per_epoch = []
        fake_score_per_epoch = []

        for real_images, _ in tqdm(train_dl):
            cur_real_score, cur_fake_score, loss_d = train_discriminator(
                gan, optimizer, criterion, real_images)
            real_score_per_epoch.append(cur_real_score)
            fake_score_per_epoch.append(cur_fake_score)
            loss_d_per_epoch.append(loss_d)

            loss_g = train_generator(gan, optimizer, criterion, real_images.size(0))
            loss_g_per_epoch.append(loss_g)

        losses_g.append(np.mean(loss_g_per_epoch))
        losses_d.append(np.mean(loss_d_per_epoch))
        real_scores.append(np.mean(real_score_per_epoch))
        fake_scores.append(np.mean(fake_score_per_epoch))

        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs,
            losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1]))

    return losses_g, losses_d, real_scores, fake_scores
