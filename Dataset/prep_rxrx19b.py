import cv2
import argparse
import numpy as np
import mahotas as mh
import multiprocessing

from pathlib import Path
from random import shuffle
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def gen_nuclei_bbox(img,
                    cent, crop,
                    n_size, n_edge,
                    n_cell=200):
    nids = [i for i in range(len(cent[0]))]
    shuffle(nids)
    for n in nids[:n_cell]:
        row, col = cent[0][n], cent[1][n]
        is_inside = n_edge < row < crop - n_edge and \
            n_edge < col < crop - n_edge
        if is_inside:
            cv2.rectangle(img,
                          (col-n_size, row-n_size),
                          (col+n_size, row+n_size),
                          (0, 0, 255),
                          2)
            cv2.circle(img, (col, row), radius=1,
                       color=(255, 0, 0), thickness=2)
    return


def gen_nuclei_data(plt_dir, out_dir,
                    n_img=-1, n_cll=200,  # n_cll = 300 if ablation study
                    size=2048, crop=1024,
                    n_size=32, n_edge=32,
                    debug=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = list(plt_dir.glob('**/*.png'))
    imgs = [im for im in imgs if '_w1.png' in str(im)]
    shuffle(imgs)
    if n_img > 0:
        imgs = imgs[:n_img]

    for img_pth in imgs:
        assert '_w1.png' in str(img_pth)
        l = (size - crop) // 2
        t = (size - crop) // 2
        # loop over staining channels
        flo = list()
        for i in range(6):
            pth = str(img_pth).replace('_w1.png', '_w{}.png'.format(i + 1))
            # center crop the image
            img = mh.imread(pth)[t:t+crop, l:l+crop]
            flo.append(img)
        flo = np.stack(flo, -1)

        # add gaussian filter
        flog = mh.gaussian_filter(flo[:, :, 0], 8)
        # find the centroid for each nuclei
        rmax = mh.regmax(flog)
        # np.array row first
        cent = np.where(rmax == rmax.max())

        img_name = img_pth.name
        nids = [i for i in range(len(cent[0]))]
        shuffle(nids)
        nval = 0
        for n in nids:
            row, col = cent[0][n], cent[1][n]
            # make sure that the image is not
            # close to the boundary
            is_inside = n_edge < row < crop - n_edge and \
                n_edge < col < crop - n_edge
            if is_inside:
                # crop the single cell image
                nuclei = flo[row-n_size:row+n_size,
                             col-n_size:col+n_size]
                # new image patch the last three channel along wid
                # col wid 256, row hei 128,
                out_img = Image.new('RGB', (n_size * 8, n_size * 4))
                out_pth = out_dir / \
                    img_name.replace('w1', '{}_{}'.format(row, col))
                for i in range(2):
                    ncl = nuclei[:, :, i * 3: (i + 1) * 3]
                    ncl = Image.fromarray(ncl.astype(np.uint8))
                    ncl = ncl.resize((n_size * 4, n_size * 4))
                    out_img.paste(ncl, (i * n_size * 4, 0))
                out_img.save(out_pth)
                nval += 1
            if nval == n_cll:
                break
        print(img_pth, len(cent[0]))

        if debug:
            flo_img = np.array(flo[:, :, :3])
            gen_nuclei_bbox(flo_img, cent, crop, n_size, n_edge)
            flo_img = Image.fromarray(flo_img.astype(np.uint8))
            new_pth = out_dir / img_name
            flo_img.save(new_pth)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='The version of processing rxrx19b for Neurips2022 submission.')
    parser.add_argument('--rxrx1_dir',
                        type=Path,
                        default=Path(
                            'Data/non_IID_old/preproc/rxrx19b/images/HUVEC-1'),
                        help="insert filepath to image")
    parser.add_argument('--rxrx1_out',
                        type=Path,
                        default=Path(
                            'Data/non_IID_old/preproc/rxrx19b_cell/images/HUVEC-1'),
                        help="insert filepath to image")
    args = parser.parse_args()

    cores = 32
    with multiprocessing.Pool(processes=cores) as pool:
        gen_args = list()
        for plt in Path(args.rxrx1_dir).iterdir():
            if plt.is_dir():
                assert 'Plate' in str(plt)
                out_plt = Path(str(plt).replace('rxrx19b', 'rxrx19b_cell'))
                gen_args.append((plt, out_plt))
        pool.starmap(gen_nuclei_data, gen_args)

    # for img_pth in Path(args.rxrx1_dir).rglob('*.png'):
    #     key = f'images/HUVEC-1/{img_pth.parent.name}/{img_pth.name}'
    #     if key not in cel_dct:
    #         print(f'{key} not in cel_dct')
    #     else:
    #         print(key, len(cel_dct[key]))
    #     del cel_dct[key]
    # with open(str(cell_pth / 'cell.json'), 'w', encoding='utf-8') as f:
    #     json.dump(cel_dct, f, ensure_ascii=False, indent=4)
