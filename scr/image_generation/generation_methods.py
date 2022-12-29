from scr.image_generation.save_methods import save_generated_images


def generate_images(gan, batch_size, batches_count, noise_vectors, path):
    for batch_num in range(batches_count):
        noise_batch = noise_vectors[batch_num * batch_size: (batch_num+1) * batch_size]
        generated_images = gan.get_generator()(noise_batch)
        save_generated_images(generated_images, batch_num, batch_size, path)
