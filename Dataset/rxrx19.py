import os
import sys
import json
import torch
import random
import warnings
import numpy as np
import pandas as pd

from cfgs import Config
from pathlib import Path
from PIL import Image, ImageFile
from itertools import combinations
from wilds.common.metrics.all_metrics import Accuracy
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.data_loaders import get_train_loader
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('.')


class rxrx19Dataset(WILDSDataset):
    _dataset_name = 'rxrx19'

    def __init__(self,
                 control,
                 cell_nam,
                 cell_num,
                 cell_chn,
                 root_dir=Path('Data/non_IID/preproc/rxrx19a_HRCE_cell'),
                 split_scheme='strat',
                 seed=0):

        self.control = control
        self.cell_nam = cell_nam
        self.cell_num = cell_num
        self.cell_chn = cell_chn
        assert self.cell_nam in str(root_dir)
        df = pd.read_csv(root_dir / 'metadata.csv')
        self.root_dir = root_dir
        random.seed(seed)

        if split_scheme in ('strat', 'abl', 'abl0', 'visual'):
            with open(f'Dataset/doc/{cell_nam}_{split_scheme}.json', 'r') as file:
                self._split_drug = json.load(file)
                self._split_dict = dict()
                print(list(self._split_drug.keys()))
        elif split_scheme == 'demo':
            with open(f'Dataset/doc/{cell_nam}_demo.json', 'r') as file:
                _dct = json.load(file)
                self._split_dict = {key: val[0] for key, val in _dct.items()}
        else:
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')

        if 'cell' in str(self.root_dir):
            self.get_cell_dict()
        else:
            self.cell_dict = None

        _filter_array = df.apply(self.exist_filepath,
                                 axis=1, result_type='expand')
        is_key = _filter_array.values[:, 0]
        cell_key = _filter_array.values[:, 1]

        df = df[is_key]
        self._split_array = self.get_splt_arry(df, split_scheme)
        self._input_array = cell_key[is_key.tolist()]

        # Labels
        self._y_array = torch.ones([len(self._input_array)])
        self._n_classes = 1
        self._y_size = 1

        meta_list = list()
        self._metadata_fields = ['experiment',
                                 'plate',
                                 'well',
                                 'site',
                                 'disease_condition',
                                 'treatment',
                                 'treatment_conc']
        for key in self._metadata_fields:
            meta_list.append(df[key].values.tolist())
        self._metadata_array = list(zip(*meta_list))

        self.get_clr_dct()
        self.get_cmp_lst()

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are
                predicted labels (LongTensor). But they can also be other model
                outputs such that prediction_fn(y_pred) are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

    def exist_filepath(self, row):
        cell_key = os.path.join('images',
                                f'{row.experiment}',
                                f'Plate{row.plate}',
                                f'{row.well}_s{row.site}_w1.png')
        has_key = self.cell_dict is not None and \
            cell_key in self.cell_dict
        return has_key, cell_key

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        # All images are in the train folder
        if 'cell' in str(self.root_dir):
            img_dir = Path(self._input_array[idx]).parent
            img_pth = self.cell_dict[self._input_array[idx]]
            img_info = list()
            for pth in img_pth:
                if pth[-4:] != '.png':
                    # occured once when . (00101110) is swtiched to ® (10101110)
                    warnings.warn(
                        f'wrong {pth} may caused by json readout error.')
                    pth = pth[:-4] + '.png'
                path = self.root_dir / img_dir / pth
                if not path.is_file():
                    warnings.warn(f'{str(path)} file does not exist.')
                    continue
                img = np.array(Image.open(path))
                img_info.append([img, pth])
            random.shuffle(img_info)
            img_list, img_path = zip(*img_info[:self.cell_num])
            img = np.stack(img_list, 0)
            if img.shape[0] == 1:
                warnings.warn(
                    f'{str(path)}_{img.shape} is the only cell extracted from a raw image.')
            col = img.shape[2] // 2
            img = np.concatenate((img[:, :, :col],
                                  img[:, :, col:]), axis=-1)
            img = img.transpose((0, 3, 1, 2))
            img = torch.from_numpy(img[:, :self.cell_chn]).contiguous()
            if img.shape[0] != self.cell_num:
                warnings.warn(
                    f'cell amounts {img.shape[0]} of {self._input_array[idx]} != {self.cell_num}')
        else:
            img_path = self.root_dir / self._input_array[idx]
            img = np.asarray(Image.open(img_path))
        return img, img_path

    def get_splt_arry(self, df, split_scheme):
        if split_scheme in ('strat', 'abl', 'abl0', 'visual'):
            assert not bool(self._split_dict)
            drug_dct = self._split_drug
        else:
            assert bool(self._split_dict)
            drug_dct = self._split_dict

        # init array with -1
        split_array = -np.ones_like(df.disease_condition.values)
        sick = 'storm-severe' if self.cell_nam == 'HUVEC' else 'Active SARS-CoV-2'
        for key, val in drug_dct.items():
            if key in self.control:
                cnt = self.control[key]
                msk = (df.disease_condition == cnt) & (df.treatment.isnull())
                if key in ('Mock', 'Irradiated', 'healthy'):
                    # healthy cell does not get treatment
                    assert (msk == (df.disease_condition == cnt)).all()

            else:
                msk = (df.disease_condition == sick) & (df.treatment == key)
                # treatment only applies on unhealthy cells
                assert (msk == (df.treatment == key)).all()

            if split_scheme in ('strat', 'abl', 'abl0', 'visual'):
                for exp in val:
                    msk_exp = msk & \
                        (df.experiment == f'{self.cell_nam}-{exp}')
                    if val[exp] is None:
                        assert not msk_exp.any()
                    else:
                        for dose, dlst in val[exp].items():
                            if key in self.control:
                                msk_val = msk_exp.values
                                cnd_nam = f'{key}-{exp}'
                                assert len(val[exp].keys()) == 1
                            else:
                                msk_dos = msk_exp & \
                                    (df.treatment_conc == float(dose))
                                msk_val = msk_dos.values
                                cnd_nam = f'{key}-{exp}-{dose}'

                            assert msk_val.any() and \
                                np.all(split_array[msk_val] == -1)
                            split_array[msk_val] = dlst[0]
                            # the 1st element of dlst
                            # is the index required for wilds subset
                            self._split_dict[cnd_nam] = dlst[0]
            else:
                # all the cond should not intersect
                assert np.all(split_array[msk.values] == -1)
                split_array[msk.values] = val
        print(self._split_dict, len(self.split_dict.keys()))
        return split_array

    def get_cell_dict(self):
        # use create_filepath as key
        # to retrieve single cell img
        _cell_json = self.root_dir / 'cell.json'
        if _cell_json.is_file():
            with open(str(_cell_json), 'r') as cfile:
                self.cell_dict = json.load(cfile)
        else:
            self.cell_dict = dict()
            exps = [f'{self.cell_nam}-1', ]
            if self.cell_nam in ('HRCE', 'VERO'):
                exps.append(f'{self.cell_nam}-2')

            for exp in exps:
                cell_dir = Path(f'images/{exp}')
                for plt in (self.root_dir / cell_dir).iterdir():
                    if plt.is_dir():
                        print(plt)
                        assert 'Plate' in str(plt)
                        for img in plt.glob('**/*.png'):
                            img_prefix = img.stem.split('_')
                            img_origin = '{}_{}_w1.png'.format(
                                img_prefix[0], img_prefix[1])
                            cell_key = str(cell_dir / plt.name / img_origin)
                            if cell_key not in self.cell_dict:
                                self.cell_dict[cell_key] = list()
                            self.cell_dict[cell_key].append(img.name)
            with open(str(_cell_json), 'w', encoding='utf-8') as f:
                json.dump(self.cell_dict, f, ensure_ascii=False, indent=4)
        print(len(list(self.cell_dict.keys())))

    def get_clr_dct(self):
        self._clr_dct = dict()
        for key in self._split_dict.keys():
            if 'Mock' in key or 'Irradiated' in key or 'healthy' in key:
                self._clr_dct[key] = (0, 0, 1)
            elif 'Infected' in key or 'storm-severe' in key:
                self._clr_dct[key] = (1, 0, 0)
            else:
                self._clr_dct[key] = (0, 1, 0)

    def get_cmp_lst(self):
        sub_keys = list(self._split_dict.keys())
        # TODO: this may not be compatabile with 'strat' split_scheme
        if 'Mock' in sub_keys[0] or 'healthy' in sub_keys[0]:
            # always compare healthy/mock to the rest
            self._cmp_lst = [[sub_keys[0], sub_keys[i]]
                             for i in range(1, len(sub_keys))]
        else:
            # this may not be useful
            self._cmp_lst = list(combinations(sub_keys, 2))


if __name__ == '__main__':
    rxrx_dct = Config().rxrx19
    for cel, bat in {'HUVEC': 'b', }.items():
        cfg = rxrx_dct[cel]
        print(cfg['control'], cel,
              cfg['cell_num'],
              cfg['cell_chn'],)
        root = Path(f'Data/non_IID/preproc/rxrx19{bat}_{cel}_cell')
        data = rxrx19Dataset(cfg['control'], cel,
                             cfg['cell_num'],
                             cfg['cell_chn'],
                             root_dir=root,
                             split_scheme='demo')
        # for ref in cfg['control']:
        #     print(ref)
        #     sub = data.get_subset(ref)
        #     loader = get_train_loader('standard', sub, 1)
        #     for num, (img, pth, meta) in enumerate(loader):
        #         print(img[1:], meta)
        #         if num == 3:
        #             break
        #     print('\n')
