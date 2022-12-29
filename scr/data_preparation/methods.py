import torchvision.transforms as tt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_dataloader(image_size, batch_size, path_to_data, stats):
    """
    Builds DataLoader for data in specified directory

    :param image_size: height and wdith of the image
    :param batch_size: batch_size of the dataloader
    :param path_to_data: directory with data
    :param stats: variance and mean for normalization

    :returns: DataLoader object
    """
    dataset = ImageFolder(path_to_data, transform=tt.Compose([
        tt.Resize(image_size),
        tt.ToTensor(),
        tt.Normalize(*stats)]))
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)


def denorm(img_tensors, stats):
    """:param stats: variance and mean for normalization"""
    return img_tensors * stats[1][0] + stats[0][0]
