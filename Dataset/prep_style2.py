import lmdb
import argparse
import multiprocessing

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from functools import partial
from torchvision import datasets


def resize_and_convert(img, size):
    if img.size != (size, size):
        img = img.resize((size, size))
    buffer = BytesIO()
    # use lossless png format
    # to save lmdb data
    img.save(buffer, format='png')
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size))

    return imgs


def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    
    if 'rxrx19' in file:
        # only allows resolution 128
        assert len(sizes) == 1 and sizes[0] == 128 
        buffer = BytesIO()
        img.save(buffer, format='png')
        out = [buffer.getvalue()]
    else:
        print(file)
        out = resize_multiple(img, sizes=sizes)

    return i, out


def prepare(env, dataset, n_worker, sizes):
    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    resize_fn = partial(resize_worker, sizes=sizes)
    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess images for model training')
    parser.add_argument('--out', type=str,
                        help='filename of the result lmdb dataset')
    parser.add_argument(
        '--size',
        type=str,
        default='128,256,512,1024',
        help='resolutions of images for the dataset',
    )
    parser.add_argument(
        '--n_worker',
        type=int,
        default=8,
        help='number of workers for preparing dataset',
    )
    parser.add_argument('path', type=str, help='path to the image dataset')

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.size.split(',')]

    print(f'Make dataset of image sizes:', ', '.join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)
    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes)
