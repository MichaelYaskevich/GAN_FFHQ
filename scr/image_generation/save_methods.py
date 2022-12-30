import os

from scr.image_generation import generate_images
from scr.visualization.image_preparation import convert_tensor_to_image


def save_generated_images(gan, batch_size, batches_count, noise_vectors, path):
    """Save all images generated from noise vectors"""
    for batch_num, batch in enumerate(generate_images(gan, batch_size, batches_count, noise_vectors)):
        save_images_batch(batch, batch_num, batch_size, path)


def save_images_batch(images, batch_num, batch_size, path):
    """Save batch of images"""
    if not os.path.exists(path):
        os.makedirs(path)
    for i, tensor in enumerate(images):
        image = convert_tensor_to_image(tensor)
        image.save(f'{path}/{batch_num * batch_size + i}.jpg')
