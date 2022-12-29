import torch
import numpy as np

from tqdm.notebook import tqdm


def train_discriminator(gan, optimizer, criterion, real_images):
    device = gan.get_device()
    # Clear discriminator gradients
    optimizer.get_discriminator_optimizer().zero_grad()

    # Pass real images through discriminator
    real_preds = gan.get_discriminator()(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = criterion.get_discriminator_loss()(real_preds, real_targets)
    cur_real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(real_images.size(0), gan.get_latent_size(), 1, 1, device=device)
    fake_images = gan.get_generator()(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = gan.get_discriminator()(fake_images)
    fake_loss = criterion.get_discriminator_loss()(fake_preds, fake_targets)
    cur_fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss_d = real_loss + fake_loss
    loss_d.backward()
    optimizer.get_discriminator_optimizer().step()

    return cur_real_score, cur_fake_score, loss_d.item()


def train_generator(gan, optimizer, criterion, batch_size):
    device = gan.get_device()
    # Clear generator gradients
    optimizer.get_generator_optimizer().zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, gan.get_latent_size(), 1, 1, device=device)
    fake_images = gan.get_generator()(latent)

    # Try to fool the discriminator
    preds = gan.get_discriminator()(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss_g = criterion.get_generator_loss()(preds, targets)

    # Update generator weights
    loss_g.backward()
    optimizer.get_generator_optimizer().step()

    return loss_g.item()


def train(train_dl, gan, criterion, optimizer, epochs):
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
