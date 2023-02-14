import sys
import json
import argparse
import numpy as np
import pandas as pd

from cfgs import Config
from pathlib import Path

sys.path.append('.')

ABL_LST = ('GS-441524', 'Remdesivir (GS-5734)', 'Baricitinib',
           'Chloroquine', 'Hydroxychloroquine Sulfate', 'Ruxolitinib', 'Tofacitinib',
           'Duvelisib', 'alpelisib', 'Crizotinib', 'Golvatinib', 'Cabozantinib',
           'Fostamatinib', 'R406')


def _rxrx19_fltr(drg_nam, df_drg):
    if df_drg.empty:
        print(f'{drg_nam} is empty, ignore.')
        return True
    if drg_nam in ('Allopregnanolone', 'atenolol-(+/-)'):
        print(f'{drg_nam} contain errors during stat calc, ignore.')
        return True
    if drg_nam in ('digoxin', 'digitoxin', 'Brilliant Green cation', 'proscillaridin-a'):
        if cel == 'HUVEC':
            print(drg_nam, 'has few cell images, ignore')
            return True
    return False


def _rxrx19_stat(df, cel_dct):
    img, cel = 0, 0
    for _, row in df.iterrows():
        key = f'images/{row.experiment}/Plate{row.plate}/{row.well}_s{row.site}_w1.png'
        if key in cel_dct:
            img += 1
            cel += len(cel_dct[key])
    if img == 0 or cel == 0:
        print(img, cel, np.unique(df.treatment.values))
    return img, cel


def _rxrx19_sort(cel, df, nan_fil, max_num):
    # sort the hit scores based on the maximum hit of two plates
    # except for HUVEC with only one plate
    if cel == 'HUVEC':
        df = df[df[f'{cel}-1'].notna()]
        df['sort'] = df[f'{cel}-1']
    else:
        df = df[(df[f'{cel}-1'].notna()) | (df[f'{cel}-2'].notna())]
        # fill in small negative value for correct maximum calc
        df[f'{cel}-1'].fillna(nan_fil, inplace=True)
        df[f'{cel}-2'].fillna(nan_fil, inplace=True)
        df['sort'] = np.maximum(df[f'{cel}-1'], df[f'{cel}-2'])

    # only keep top 'max_num' drugs
    df_sort = df.sort_values('sort', ascending=False)
    if max_num is not None:
        df_sort = df_sort.head(max_num)
    return df_sort


def _rxrx19_app(df_sort, df, cel):
    for c in cel:
        df_sort = df_sort.append(df[df['name'] == c])
    return df_sort


def rxrx19_hit_split(csv_dir):
    # split HUMAN.csv to HRCE and HUVEC
    df = pd.read_csv(str(csv_dir / 'HUMAN.csv'))

    hrce_col = ['compound_id', 'name', 'HRCE-1', 'HRCE-2', 'smiles']
    df_hrce = df[hrce_col]
    df_hrce.to_csv(str(csv_dir / 'HRCE_hit.csv'), index=False)

    huvec_col = ['compound_id', 'name', 'HUVEC-1', 'smiles']
    df_huvec = df[huvec_col]
    df_huvec.to_csv(str(csv_dir / 'HUVEC_hit.csv'), index=False)


def rxrx19_hit_json(cel,
                    control,
                    csv_dir,
                    max_num,
                    add_cel=None,
                    nan_fil=-10e8):
    drg_dct = dict()
    # fill in hit socres (None) for control groups
    for cont in control:
        drg_dct[cont] = [None, None]

    df = pd.read_csv(str(csv_dir / f'{cel}_hit.csv'))
    df_sort = _rxrx19_sort(cel, df, nan_fil, max_num)

    if add_cel and max_num is not None:
        df_sort = _rxrx19_app(df_sort, df, add_cel)
        df_sort[f'{cel}-1'].fillna(nan_fil, inplace=True)
        if cel != 'HUVEC':
            df_sort[f'{cel}-2'].fillna(nan_fil, inplace=True)

    df_meta = pd.read_csv(str(csv_dir / f'{cel}_meta.csv'))
    for _, row in df_sort.iterrows():
        drg_nam = row['name']
        df_drg = df_meta[df_meta.treatment == drg_nam]
        if _rxrx19_fltr(drg_nam, df_drg):
            continue
        assert drg_nam not in drg_dct
        drg_dct[drg_nam] = list()
        for exp in (1, 2):
            if (exp == 2 and cel == 'HUVEC') or row[f'{cel}-{exp}'] == nan_fil:
                drg_dct[drg_nam].append(None)
            else:
                drg_dct[drg_nam].append(row[f'{cel}-{exp}'])

    cel_jsn = str(csv_dir / f'{cel}.json')
    with open(str(cel_jsn), 'w', encoding='utf-8') as f:
        json.dump(drg_dct, f, ensure_ascii=False, indent=4)


def rxrx19_strat_json(cel,
                      control,
                      csv_dir,
                      jsn_pth,
                      max_num,
                      add_cel=None,
                      nan_fil=-10e8):

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
                img_num, cel_num = _rxrx19_stat(df_exp, cel_dct)
                # acc: index for wildssubset,
                # img_num: the amount of images
                # cel_num: the amount of cells
                # add a pseudo dose -1
                drg_dct[key][e] = dict()
                drg_dct[key][e][-1] = (acc, img_num, cel_num)
                acc += 1
    print(drg_dct, '\n')

    df = pd.read_csv(str(csv_dir / f'{cel}_hit.csv'))
    df_sort = _rxrx19_sort(cel, df, nan_fil, max_num)
    if add_cel and max_num is not None:
        df_sort = _rxrx19_app(df_sort, df, add_cel)
    # drug stratification
    for _, row in df_sort.iterrows():
        drg_nam = row['name']
        df_drg = df_meta[df_meta.treatment == drg_nam]
        if _rxrx19_fltr(drg_nam, df_drg):
            continue
        assert drg_nam not in drg_dct, drg_nam
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
                    img_num, cel_num = _rxrx19_stat(df_dos, cel_dct)
                    if cel_num <= 100:
                        print(drg_nam, e, dos, img_num)
                    drg_dct[drg_nam][e][dos] = (acc, img_num, cel_num)
                    acc += 1

    cel_jsn = str(csv_dir / f'{cel}_strat.json')
    with open(str(cel_jsn), 'w', encoding='utf-8') as f:
        json.dump(drg_dct, f, ensure_ascii=False, indent=4)


def rxrx19_visual_json(cel,
                       control,
                       csv_dir,
                       jsn_pth,
                       add_drg):

    with open(str(jsn_pth), 'r') as cfile:
        cel_dct = json.load(cfile)
    df_meta = pd.read_csv(str(csv_dir / f'{cel}_meta.csv'))

    acc, drg_dct = 0, dict()
    # control group stratification
    for key, cnt in control.items():
        drg_dct[key] = dict()
        df_cnt = df_meta[(df_meta.disease_condition == cnt)
                         & (df_meta.treatment.isnull())]

        assert not df_cnt.empty
        img_num, cel_num = _rxrx19_stat(df_cnt, cel_dct)
        # acc: index for wildssubset,
        # img_num: the amount of images
        # cel_num: the amount of cells
        drg_dct[key] = (acc, img_num, cel_num)
        acc += 1
    print(drg_dct, '\n')

    for drg in add_drg:
        df_drg = df_meta[df_meta.treatment == drg]
        assert not df_drg.empty
        img_num, cel_num = _rxrx19_stat(df_drg, cel_dct)
        drg_dct[drg] = (acc, img_num, cel_num)
        acc += 1

    cel_jsn = str(csv_dir / f'{cel}_visual.json')
    with open(str(cel_jsn), 'w', encoding='utf-8') as f:
        json.dump(drg_dct, f, ensure_ascii=False, indent=4)


def rxrx19_json_test(cel,
                     control,
                     csv_dir):
    with open(str(csv_dir / f'{cel}.json'), 'r') as f:
        hit_dct = json.load(f)
    with open(str(csv_dir / f'{cel}_strat.json'), 'r') as f:
        strat_dct = json.load(f)

    acc = 6 if cel != 'HUVEC' else 2
    for cnd in strat_dct:
        if cnd not in control:
            for e in (1, 2):
                if hit_dct[cnd][e - 1] is None:
                    assert strat_dct[cnd][str(e)] is None
                else:
                    for key, val in strat_dct[cnd][str(e)].items():
                        assert val[0] == acc
                        acc += 1


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
        '--max_num',
        type=int,
        default=200,
        help='maximum amount of examined drugs')

    args = parser.parse_args()

    rxrx_cfg = Config()

    # for cel, bat in {'VERO': 'a', 'HRCE': 'a', 'HUVEC': 'b'}.items():
    #     add_cel = None
    #     control = rxrx_cfg.rxrx19[cel]['control']
    #     rxrx19_hit_json(cel, control,
    #                     args.csv_dir,
    #                     None,
    #                     add_cel)
    #     jsn_pth = str(args.jsn_dir / f'rxrx19{bat}_{cel}_cell' / 'cell.json')
    #     rxrx19_strat_json(cel, control,
    #                       args.csv_dir,
    #                       jsn_pth,
    #                       None,
    #                       add_cel)
    #     rxrx19_json_test(cel, control,
    #                      args.csv_dir)

    # for cel in ['HRCE', 'VERO', 'HUVEC']:
    #     print(cel)
    #     df_hit = pd.read_csv(str(args.csv_dir / f'{cel}_hit.csv'))
    #     df_meta = pd.read_csv(str(args.csv_dir / f'{cel}_meta.csv'))
    #     for rid, row in df_hit.iterrows():
    #         drug = row['name']
    #         df_drug = df_meta[df_meta.treatment == drug]
    #         df_drug_1 = df_drug[df_drug.experiment == f'{cel}-1']
    #         df_drug_2 = df_drug[df_drug.experiment == f'{cel}-2']
    #         if not (df_drug_1.empty or df_drug_2.empty):
    #             print(cel, drug)
    #             df_dose_1 = np.unique(df_drug_1.treatment_conc).tolist()
    #             df_dose_1.sort()
    #             df_dose_2 = np.unique(df_drug_2.treatment_conc).tolist()
    #             df_dose_2.sort()
    #             if df_dose_1 != df_dose_2:
    #                 print(drug, df_dose_1, df_dose_2)

    for cel, bat in {'HUVEC': 'b', }.items():
        if cel == 'VERO':
            add_drg = ('GS-441524', 'Remdesivir (GS-5734)',
                       'Chloroquine', 'Hydroxychloroquine Sulfate')
        elif cel == 'HRCE':
            add_drg = ('GS-441524', 'Remdesivir (GS-5734)',
                       'Chloroquine', 'Hydroxychloroquine Sulfate',
                       'Baricitinib', 'Ruxolitinib')
        elif cel == 'HUVEC':
            add_drg = ABL_LST

        control = rxrx_cfg.rxrx19[cel]['control']
        jsn_pth = str(args.jsn_dir / f'rxrx19{bat}_{cel}_cell' / 'cell.json')
        rxrx19_visual_json(cel,
                           control,
                           args.csv_dir,
                           jsn_pth,
                           add_drg)
