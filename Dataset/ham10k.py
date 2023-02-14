
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from PIL import Image, ImageFile
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.metrics.all_metrics import Accuracy

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ham10kDataset(WILDSDataset):
    _dataset_name = 'ham10k'

    def __init__(self,
                 root_dir=Path('Data/non_IID/preproc/ham10k_tiny'),
                 split_scheme='baseline'):

        df = pd.read_csv(root_dir / 'metadata.csv')
        self.root_dir = root_dir
        # Splits
        if split_scheme in ('baseline', 'visual'):
            self._split_dict = {
                'nv': 6705,
                'mel': 1113,
                'bcc': 514,
                'bkl': 1099,
                'df': 115,
                'vasc': 142,
                'akiec': 327
            }
            self._split_array = df.dx.apply(self._split_dict.get).values
        else:
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')

        # Filenames
        def create_filepath(row):
            filepath = os.path.join('images',
                                    f'{row.image_id}.png')
            return filepath
        self._input_array = df.apply(create_filepath, axis=1).values

        # Labels
        self._y_array = torch.ones([len(self._input_array)])
        self._n_classes = 1
        self._y_size = 1

        meta_list = list()
        self._metadata_fields = [
            'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'dataset']
        for key in self._metadata_fields:
            meta_list.append(df[key].values.tolist())
        self._metadata_array = list(zip(*meta_list))

        self.get_clr_dct()
        self.get_cmp_lst(split_scheme)

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

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        # All images are in the train folder
        img_path = self.root_dir / self._input_array[idx]
        img = np.array(Image.open(img_path))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).contiguous()
        # img = img.to(torch.float32) / 127.5 - 1
        return img

    def get_clr_dct(self):
        self._clr_dct = {
            'mel': [(1, 0, 0)],
            'mel_0.1': [(1, 0, 0)],
            # 'mel_0.2': [(1, 0, 0)],
            'mel_0.3': [(1, 0, 0)],
            'nv': [(0, 0, 1)],
            'nv_0.1': [(0, 0, 1)],
            # 'nv_0.2': [(0, 0, 1)],
            'nv_0.3': [(0, 0, 1)],
            'old': [(1, 0, 0)],
            'young': [(0, 0, 1)],
            'male': [(1, 0, 0)],
            'female': [(0, 0, 1)],
            'bcc': [(1, 0.5, 0)],
            'bkl': [(0, 0, 1)],
            'df': [(0.5, 1, 0)],
            'vasc': [(1, 0, 1)]}

    def get_cmp_lst(self, split_scheme):
        sub_keys = list(self._split_dict.keys())
        if 'toy' in split_scheme or 'baseline' in split_scheme:
            self._cmp_lst = [[sub_keys[0], sub_keys[i]]
                             for i in range(1, len(sub_keys))]
        elif 'rebuttal' in split_scheme:
            self._cmp_lst = [[sub_keys[0], sub_keys[i]]
                             for i in range(1, len(sub_keys))]
            if sub_keys[1] in ('mel', 'nv') and len(sub_keys) > 2:
                _cmp_lst1 = [[sub_keys[1], sub_keys[i]]
                             for i in range(2, len(sub_keys))]
                self._cmp_lst.extend(_cmp_lst1)
        else:
            # _cmp_lst may not be very useful
            # if more than 2 sub_keys
            self._cmp_lst = list(combinations(sub_keys, 2))


if __name__ == '__main__':
    data = ham10kDataset(split_scheme='age')
