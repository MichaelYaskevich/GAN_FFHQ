import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

from scr.data_preparation import DeviceDataLoader


def get_leave_one_out_1nn_cls_accuracy(
        real_images_dl: DeviceDataLoader, fake_images_dl: DeviceDataLoader,
        feature_extractor) -> float:
    """Counts leave one out 1nn (k nearest neighbors with k == 1) classifier accuracy"""
    batch_size = real_images_dl.get_batch_size()

    fake_labels = torch.zeros(batch_size, 1)
    real_labels = torch.ones(batch_size, 1)
    labels = torch.cat((fake_labels, real_labels), 0).view(-1)

    fake_iter = iter(fake_images_dl)
    real_iter = iter(real_images_dl)
    loss = 0

    for i in range(len(real_images_dl) - 1):
        fake_batch = get_batch_of_features(fake_iter, feature_extractor)
        real_batch = get_batch_of_features(real_iter, feature_extractor)

        data = torch.cat((fake_batch, real_batch), 0)
        loss += get_batch_loss(data, labels)

    return loss / ((len(real_images_dl) - 1) * 2 * batch_size)


def get_batch_of_features(batch_iter, feature_extractor) -> torch.Tensor:
    batch, _ = next(batch_iter)
    batch = feature_extractor(batch)
    return batch.view(batch.size(0), -1).cpu().detach()


def get_batch_loss(data, labels) -> float:
    loo = LeaveOneOut()
    batch_loss = 0
    for train, test in loo.split(data):
        cls = KNeighborsClassifier(n_neighbors=1)
        cls.fit(data[train], labels[train])
        pred = cls.predict(data[test])
        batch_loss += abs(pred.item()-labels[test].item())
    return batch_loss
