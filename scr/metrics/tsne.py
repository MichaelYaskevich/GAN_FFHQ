import numpy as np
import torch

from sklearn.manifold import TSNE

from scr.metrics.loo_1nn_cls_accuracy import get_batch_of_features


def tsne_for_batch(real_images_dl, fake_images_dl, feature_extractor) -> np.ndarray:
    fake_batch = get_batch_of_features(iter(fake_images_dl), feature_extractor)
    real_batch = get_batch_of_features(iter(real_images_dl), feature_extractor)

    data = torch.cat((fake_batch, real_batch), 0)
    return TSNE(n_components=2, learning_rate='auto',
                init='random').fit_transform(data)
