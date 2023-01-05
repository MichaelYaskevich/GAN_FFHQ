def to_device(data, device):
    """
    Move tensor(s) to chosen device

    :param data: tensor(s)
    :param device: chosen device
    """

    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """
    Wrap a dataloader to yield batches on specified device
    """

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        self._batch_size = dl.batch_size

    def __iter__(self):
        """Yield a batch on specified device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    def get_batch_size(self):
        """Number of elements in each bach except the last one"""
        return self._batch_size
