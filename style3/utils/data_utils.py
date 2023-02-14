"""
Code adopted from pix2pixHD (https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py)
"""
import torch
from pathlib import Path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename: Path):
    return any(str(filename).endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir: Path):
    images = []
    assert dir.is_dir(), '%s is not a valid directory' % dir
    for fname in dir.glob("**/*"):
        if is_image_file(fname):
            images.append(fname)
    print('size of dataset: {}'.format(len(images)))
    return images

def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32).to(start) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]

    return out