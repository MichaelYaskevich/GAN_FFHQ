import os

from scr.visualization.image_preparation import tensor_to_image


def save_generated_images(generated_images, batch_num, batch_size, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, tensor in enumerate(generated_images):
        image = tensor_to_image(tensor)
        image.save(f'{path}/{batch_num * batch_size + i}.jpg')
