import sys
import cv2
import math
import random
import argparse
import numpy as np
import mahotas as mh
import multiprocessing

from cfgs import Config
from pathlib import Path
from random import shuffle
from argparse import Namespace
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('.')


def gen_nuclei_bbox(img,
                    cthr, cent,
                    crop, nids,
                    n_size, n_edge, n_cell):
    nval = 0
    for n in nids:
        row, col = cent[0][n], cent[1][n]
        is_inside = n_edge < row < crop - n_edge and \
            n_edge < col < crop - n_edge
        if is_inside and img[row, col, 0] > cthr:
            cv2.rectangle(img,
                          (col-n_size, row-n_size),
                          (col+n_size, row+n_size),
                          (0, 0, 255),
                          2)
            cv2.circle(img, (col, row), radius=1,
                       color=(255, 0, 0), thickness=2)
            nval += 1
        if nval == n_cell:
            break
    return


def gen_nuclei_data(cfg,
                    plt_dir,
                    out_dir,
                    isabl=False,
                    debug=False,
                    corner=False):

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = list(plt_dir.glob('**/*.png'))
    imgs = [im for im in imgs if '_w1.png' in str(im)]
    shuffle(imgs)

    for img_pth in imgs:
        assert '_w1.png' in str(img_pth)
        l = (cfg.size - cfg.crop) // 2
        t = (cfg.size - cfg.crop) // 2
        # loop over staining channels
        flo = list()
        for i in range(cfg.cell_chn):
            pth = str(img_pth).replace('_w1.png', '_w{}.png'.format(i + 1))
            # center crop the image
            img = mh.imread(pth)[t:t+cfg.crop, l:l+cfg.crop]
            flo.append(img)
        if cfg.cell_chn == 5:
            flo.append(np.zeros_like(flo[-1]))
        flo = np.stack(flo, -1)

        # add gaussian filter
        flog = mh.gaussian_filter(flo[:, :, 0], cfg.sigma)
        # find the centroid for each nuclei
        rmax = mh.regmax(flog)
        # np.array row first
        cent = np.where(rmax == rmax.max())

        if corner:
            # This can be useful to filter out
            # images with lots of dead cells
            # HRCE
            if len(cent[0]) < 500 and 'HRCE' in str(out_dir):
                continue

            # # VERO
            if len(cent[0]) > 200 and 'VERO' in str(out_dir):
                continue

            # HUVEC
            if len(cent[0]) < 1000 and 'HUVEC' in str(out_dir):
                continue

        # calc the dna color intensity threshold
        # to filter out false positive dna
        cdna = flo[:, :, 0]
        cval = np.sort(cdna[rmax == rmax.max()].flatten())
        if len(cval) <= 1:
            continue
        clen = math.ceil(len(cval) * 0.1)
        # if ablation, then make flo[row, col, 0] > cthr True
        cthr = max(cval[clen], cfg.thres) if not isabl else 0

        img_name = img_pth.name
        nids = [i for i in range(len(cent[0]))]
        shuffle(nids)
        nval = 0
        if not debug:
            for n in nids:
                row, col = cent[0][n], cent[1][n]
                # make sure that the image is not
                # close to the boundary
                is_inside = cfg.cell_edg < row < cfg.crop - cfg.cell_edg and \
                    cfg.cell_edg < col < cfg.crop - cfg.cell_edg
                if is_inside and flo[row, col, 0] > cthr:
                    # crop the single cell image
                    nuclei = flo[row-cfg.cell_dim:row+cfg.cell_dim,
                                 col-cfg.cell_dim:col+cfg.cell_dim]
                    # new image patch the last three channel along wid
                    # col wid 256, row hei 128,
                    out_img = Image.new('RGB',
                                        (cfg.cell_dim * 8, cfg.cell_dim * 4))
                    out_pth = out_dir / \
                        img_name.replace('w1', '{}_{}'.format(row, col))
                    for i in range(2):
                        ncl = nuclei[:, :, i * 3: (i + 1) * 3]
                        ncl = Image.fromarray(ncl.astype(np.uint8))
                        ncl = ncl.resize((cfg.cell_dim * 4, cfg.cell_dim * 4))
                        out_img.paste(ncl, (i * cfg.cell_dim * 4, 0))
                    out_img.save(out_pth)
                    nval += 1
                if nval == cfg.cell_num:
                    break
        else:
            flo_img = np.array(flo[:, :, :3])
            gen_nuclei_bbox(flo_img,
                            cthr, cent,
                            cfg.crop, nids,
                            cfg.cell_dim, cfg.cell_edg, cfg.cell_num)

            flo_im = Image.fromarray(flo_img[:, :, 0].astype(np.uint8))
            new_pth = out_dir / img_name
            flo_im.save(new_pth)

            flo_im = Image.fromarray(flo_img[:, :, :3].astype(np.uint8))
            new_pth = out_dir / img_name.replace('w1', 'w11')
            flo_im.save(new_pth)
        print(img_pth, len(cval), clen, cval[clen], cfg.thres, nval)


if __name__ == '__main__':
    np.random.seed(10)
    random.seed(10)

    parser = argparse.ArgumentParser(
        description='The updated version of processing rxrx19{a,b} datasets.')
    parser.add_argument('--root',
                        type=Path,
                        default=Path(
                            'Data/non_IID/preproc/'),
                        help="insert filepath to image")
    parser.add_argument('--rxrx_name',
                        type=str,
                        choices=('rxrx19a', 'rxrx19b'),
                        help='the name of biomedical datasets used for application studies')
    parser.add_argument('--rxrx_cell',
                        type=str,
                        choices=('VERO', 'HRCE', 'HUVEC'),
                        help='the name of cell types')
    parser.add_argument('--is_abl',
                        action='store_true',
                        help='whether to output cells for ablation study')
    parser.add_argument('--is_debug',
                        action='store_true',
                        help='whether to visualize the cropped cells in the raw image with bbox')
    parser.add_argument('--is_corner',
                        action='store_true',
                        help='whether to only visualize the cropped cells in the corner-case raw image with bbox')
    args = parser.parse_args()

    rxrx_nm = args.rxrx_name
    if args.is_abl:
        rxrx_nm += f'_{args.rxrx_cell}_abl'
    rxrx_dir = args.root / rxrx_nm / 'images'
    rxrx_dct = Config().rxrx19[args.rxrx_cell]
    rxrx_cfg = Namespace(**rxrx_dct)
    core_num = 16
    with multiprocessing.Pool(processes=core_num) as pool:
        gen_args = list()
        for ctype in Path(rxrx_dir).iterdir():
            if ctype.is_dir() and args.rxrx_cell in str(ctype):
                for plt in ctype.iterdir():
                    if plt.is_dir():
                        print(plt)
                        assert 'Plate' in str(plt)
                        out_plt = Path(str(plt).replace(
                            args.rxrx_name, args.rxrx_name + '_cell'))
                        gen_args.append((rxrx_cfg, plt, out_plt,
                                         args.is_abl, args.is_debug, args.is_corner))
        pool.starmap(gen_nuclei_data, gen_args)
