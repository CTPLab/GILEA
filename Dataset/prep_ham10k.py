import argparse
import multiprocessing

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import partial


def prepare_ham10k(imgs,
                   size,
                   new_size,
                   n_worker):

    resize_fn = partial(resize_worker,
                        size=size,
                        new_size=new_size)

    with multiprocessing.Pool(n_worker) as pool:
        for pth, img in tqdm(pool.imap(resize_fn, imgs)):
            img.save(pth, 'PNG')


def resize_worker(file, size, new_size):
    old_pth, new_pth = file
    new_img = Image.open(old_pth).resize((size, size))
    new_img = new_img.resize((new_size, new_size))

    return new_pth, new_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess ham10k images')

    parser.add_argument(
        '--root',
        type=str,
        help='path to the raw images')

    parser.add_argument(
        '--size',
        type=int,
        default=64,
        help='downsampled image resolution')

    parser.add_argument(
        '--n_worker',
        type=int,
        default=16,
        help='number of workers for preparing dataset')

    args = parser.parse_args()

    for im_dir in Path(args.root).iterdir():
        if im_dir.is_dir():
            out_dir = Path(str(im_dir).replace('ham10k', 'ham10k_tiny'))
            out_dir.mkdir(parents=True, exist_ok=True)
            print(im_dir, out_dir)
            imgs = list()
            for im_pth in im_dir.glob('*.jpg'):
                im_pth_new = out_dir / (im_pth.stem + '.png')
                imgs.append((im_pth, im_pth_new))
            print(len(imgs))

            prepare_ham10k(imgs,
                           args.size,
                           args.size * 2,
                           args.n_worker)
