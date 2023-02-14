import os
import json
import torch
import zipfile
import argparse
import PIL.Image
import PIL.ImageFile
import numpy as np
try:
    import pyspng
except ImportError:
    pyspng = None

from pathlib import Path
from typing import Union
from torchvision import transforms
# ----------------------------------------------------------------------------

# stylegan3


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,                   # Name of the dataset.
                 raw_shape,              # Shape of the raw image data (NCHW).
                 # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 max_size=None,
                 # Enable conditioning labels? False = label dimension is zero.
                 use_labels=False,
                 # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 xflip=False,
                 # Random seed to use when applying max_size.
                 random_seed=0,
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate(
                [self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros(
                    [self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

# ----------------------------------------------------------------------------

# This function borrows from stylegan3
# the default collate function of pytorch dataloder
# will automatically convert numpy to tensor
# even without tranform


class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,                   # Path to directory or zip.
                 # Ensure specific resolution, None = highest available.
                 resolution=None,
                 # Additional arguments for the Dataset base class.
                 **super_kwargs,
                 ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(
                root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(
            fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + \
            list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
                if 'rxrx19b' in self._path:
                    col = image.shape[1] // 2
                    image = np.concatenate((image[:, :col],
                                            image[:, col:]), axis=-1)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')]
                  for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

# ----------------------------------------------------------------------------


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]


def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION  # type: ignore


# stylegan3-editing
class ImagesDataset(Dataset):

    def __init__(self, pdir, paths, target_transform=None, source_transform=None):
        self.pdir = pdir
        self.source_paths = paths
        self.target_paths = paths
        self.source_transform = source_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.pdir / self.source_paths[index]
        to_path = self.pdir / self.target_paths[index]

        from_im = np.array(PIL.Image.open(from_path))
        to_im = np.array(PIL.Image.open(to_path))
        if 'rxrx19b' in str(from_path):
            col = from_im.shape[1] // 2
            from_im = np.concatenate((from_im[:, :col],
                                      from_im[:, col:]), axis=-1)
            to_im = np.concatenate((to_im[:, :col],
                                    to_im[:, col:]), axis=-1)

        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess images for model training")
    parser.add_argument('--img_dir', type=str,
                        help="dir of the image dataset")
    parser.add_argument('--resolution', type=int, help="image resolution")

    args = parser.parse_args()
    size = args.resolution

    PIL.Image.init()
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

    size = args.resolution
    img_gen = ImageFolderDataset(args.img_dir, resolution=size)
    img_pth = img_gen._image_fnames
    print(len(img_pth))
    img_gen = torch.utils.data.DataLoader(
        dataset=img_gen, batch_size=1, shuffle=False)

    im_chn = 6 if 'rxrx19b' in args.img_dir else 3
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * im_chn, [0.5] * im_chn, inplace=True),
        ]
    )
    img_inv = ImagesDataset(Path(args.img_dir), img_pth, transform)

    for idx, im0 in enumerate(img_gen):
        if (idx + 1) % 1000 == 0:
            print(idx)
        im0 = im0[0].to(torch.float32) / 127.5 - 1
        im1 = img_inv[idx][0]
        torch.allclose(im0, im1)
