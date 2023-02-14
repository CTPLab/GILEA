import os
import sys
import json
import torch
import warnings
import argparse
import numpy as np
import pandas as pd

from cfgs import Config
from pathlib import Path
from PIL import Image, ImageFile
from itertools import combinations
from wilds.common.metrics.all_metrics import Accuracy
from wilds.datasets.wilds_dataset import WILDSDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('.')

ABL_LST = ('GS-441524', 'Remdesivir (GS-5734)', 'Baricitinib',
           'Chloroquine', 'Hydroxychloroquine Sulfate', 'Ruxolitinib', 'Tofacitinib',
           'Duvelisib', 'alpelisib', 'Crizotinib', 'Golvatinib', 'Cabozantinib',
           'Fostamatinib', 'R406')

ABL_DCT = {'VERO': ['GS-441524', 'Remdesivir (GS-5734)',
                    'Chloroquine', 'Hydroxychloroquine Sulfate'],
           'HRCE': list(ABL_LST),
           'HUVEC': list(ABL_LST)}


def _rxrx19_stat(df, cel_dct, cel_num):
    img, cel, tot = 0, 0, 0
    for _, row in df.iterrows():
        key = f'images/{row.experiment}/Plate{row.plate}/{row.well}_s{row.site}_w1.png'
        if key in cel_dct:
            img += 1
            cel += min(len(cel_dct[key]), cel_num)
            tot += len(cel_dct[key])
    if img == 0 or cel == 0:
        print(img, cel, np.unique(df.treatment.values))
    return img, cel, tot


def get_img(df, chn):
    img_lst = []
    for _, row in df.iterrows():
        for c in range(chn):
            img = f'images/{row.experiment}/Plate{row.plate}/{row.well}_s{row.site}_w{c + 1}.png'
            img_lst.append(img)
    # print(img_lst, '\n\n')
    return img_lst


def create_subset_folder(cel, csv_dir, control):
    df_meta = pd.read_csv(str(csv_dir / f'{cel}_meta.csv'))
    chn = 6 if cel == 'HUVEC' else 5

    img_lst = []
    for drg in ABL_LST:
        df_drg = df_meta[df_meta.treatment == drg]
        if not df_drg.empty:
            img_lst += get_img(df_drg, chn)

    for key, val in control.items():
        df_cnt = df_meta[(df_meta.disease_condition == val)
                         & (df_meta.treatment.isnull())]
        if not df_cnt.empty:
            img_lst += get_img(df_cnt, chn)
    return img_lst


def test_subset_folder(cel, csv_dir, sym_dir, control):
    df_meta = pd.read_csv(str(csv_dir / f'{cel}_meta.csv'))
    chn = 6 if cel == 'HUVEC' else 5
    for _, row in df_meta.iterrows():
        if (pd.isnull(row.treatment) and row.disease_condition in control) or \
           (row.treatment in ABL_LST):
            for c in range(chn):
                img = f'images/{row.experiment}/Plate{row.plate}/{row.well}_s{row.site}_w{c + 1}.png'
                sym_pth = sym_dir / img
                assert sym_pth.is_symlink() and sym_pth.exists(), \
                    f'{sym_dir / img}'


def rxrx19_abl_json(cel,
                    cel_lim,
                    control,
                    csv_dir,
                    jsn_pth):

    with open(str(jsn_pth), 'r') as cfile:
        cel_dct = json.load(cfile)
    df_meta = pd.read_csv(str(csv_dir / f'{cel}_meta.csv'))

    acc, exp, drg_dct = 0, (1, 2), dict()
    # control group stratification
    for key, cnt in control.items():
        drg_dct[key] = dict()
        df_cnt = df_meta[(df_meta.disease_condition == cnt)
                         & (df_meta.treatment.isnull())]
        for e in exp:
            df_exp = df_cnt[df_cnt.experiment == f'{cel}-{e}']
            if df_exp.empty:
                drg_dct[key][e] = None
            else:
                img_num, cel_num, cel_tot = _rxrx19_stat(
                    df_exp, cel_dct, cel_lim)
                # acc: index for wildssubset,
                # img_num: the amount of images
                # cel_num: the amount of cells
                # add a pseudo dose -1
                drg_dct[key][e] = dict()
                drg_dct[key][e][-1] = (acc, img_num, cel_num)
                acc += 1
                print(key, e, cel_num, cel_tot)

    for drg_nam in ABL_DCT[cel]:
        df_drg = df_meta[df_meta.treatment == drg_nam]
        # 'Tofacitinib' and 'Golvatinib' not in HRCE
        # use the following cond to avoid add empty drug in json
        if df_drg.empty:
            print(drg_nam, 'empty')
            continue
        drg_dct[drg_nam] = dict()
        for e in exp:
            df_exp = df_drg[df_drg.experiment == f'{cel}-{e}']
            if df_exp.empty:
                drg_dct[drg_nam][e] = None
            else:
                drg_dct[drg_nam][e] = dict()
                dose = np.unique(df_exp.treatment_conc).tolist()
                for dos in dose:
                    df_dos = df_exp[df_exp.treatment_conc == dos]
                    img_num, cel_num, cel_tot = _rxrx19_stat(
                        df_dos, cel_dct, cel_lim)
                    drg_dct[drg_nam][e][dos] = (acc, img_num, cel_num)
                    acc += 1
                    print(drg_nam, e, dos, cel_num, cel_tot)

    cel_jsn = str(csv_dir / f'{cel}_abl.json')
    with open(str(cel_jsn), 'w', encoding='utf-8') as f:
        json.dump(drg_dct, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare multiple subsets for rxrx19 experiments')

    parser.add_argument(
        '--csv_dir',
        type=Path,
        default=Path('Dataset/doc/'),
        help='path to the rxrx19 hitscores')

    parser.add_argument(
        '--jsn_dir',
        type=Path,
        default=Path('Data/non_IID/preproc/'),
        help='path to the cell json file')

    parser.add_argument(
        '--img_dir',
        type=Path,
        default=Path('Data/non_IID/preproc/'),
        help='path to the cell json file')

    args = parser.parse_args()
    rxrx_cfg = Config()

    for cel, bat in {'HUVEC': 'b', }.items():
        control = rxrx_cfg.rxrx19[cel]['control']
        cel_lim = rxrx_cfg.rxrx19[cel]['cell_num']
        print(cel, control)
        img_lst = create_subset_folder(cel, args.csv_dir, control)
        for img in img_lst:
            src = args.img_dir / f'rxrx19{bat}' / img
            dst = args.img_dir / f'rxrx19{bat}_{cel}_abl' / img
            if not dst.parent.is_dir():
                dst.parent.mkdir(exist_ok=True, parents=True)
            os.symlink(str(src), str(dst))

        sym_dir = args.img_dir / f'rxrx19{bat}_{cel}_abl'
        test_subset_folder(cel, args.csv_dir, sym_dir, control)

        # jsn_pth = str(
        #     args.jsn_dir / f'rxrx19{bat}_{cel}_cell_abl' / 'cell.json')
        # rxrx19_abl_json(cel, cel_lim,
        #                 control,
        #                 args.csv_dir,
        #                 jsn_pth)
