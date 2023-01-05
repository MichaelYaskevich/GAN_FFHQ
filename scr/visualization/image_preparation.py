import PIL
import numpy as np

from scr.data_preparation import denorm


def convert_tensor_to_image(tensor, stats):
    tensor = tensor.permute(1, 2, 0).cpu().detach()
    tensor = denorm(tensor, stats)
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    return PIL.Image.fromarray(tensor)
