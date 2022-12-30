import PIL
import numpy as np


def convert_tensor_to_image(tensor):
    tensor = tensor.permute(1, 2, 0).cpu().detach()
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    return PIL.Image.fromarray(tensor)
