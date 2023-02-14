
import torch
import lmdb
import argparse
import numpy as np

from io import BytesIO
from PIL import Image, ImageFile
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms

# stylegan2-pytorch


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=128):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform
        self.path = path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = np.asarray(img)
        if 'rxrx19b' in self.path:
            col = img.shape[1] // 2
            img = np.concatenate((img[:, :col],
                                  img[:, col:]), axis=-1)
        img = self.transform(img)

        return img

# restyle-encoder


class ImagesDataset(Dataset):

    def __init__(self, paths, target_transform=None, source_transform=None):
        self.source_paths = paths
        self.target_paths = paths
        self.source_transform = source_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        to_path = self.target_paths[index]

        from_im = np.array(Image.open(from_path))
        to_im = np.array(Image.open(to_path))
        if 'rxrx19b' in from_path:
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
    parser.add_argument('--imgs_dir', type=str,
                        help="dir of the image dataset")
    parser.add_argument('--lmdb_dir', type=str, help="dir if the lmdb dataset")
    parser.add_argument('--resolution', type=int, help="image resolution")

    args = parser.parse_args()
    size = args.resolution

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    im_chn = 6 if 'rxrx19b' in args.imgs_dir else 3
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * im_chn, [0.5] * im_chn, inplace=True),
        ]
    )
    img_lmdb = MultiResolutionDataset(args.lmdb_dir, transform, size)

    dataset = datasets.ImageFolder(args.imgs_dir)
    img_pths = sorted(dataset.imgs,
                      key=lambda x: x[0])
    img_pths = [im[0] for im in img_pths]
    img_invs = ImagesDataset(img_pths, transform)

    for idx in range(len(img_invs)):
        if (idx + 1) % 1000 == 0:
            print(idx)

        im0 = img_lmdb[idx]
        im1 = img_invs[idx][0]
        torch.allclose(im0, im1)
