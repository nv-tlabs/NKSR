import numpy as np

import dataset
from dataset.base import Dataset, RandomSafeDataset
from dataset.base import DatasetSpec as DS
import multiprocessing
from pycg import exp


class CombinedDataset(Dataset):
    def __init__(self, config_list, spec, hparams=None, **kwargs):
        self.spec = spec
        self.hparams = hparams
        self.datasets = []
        self.config_list = config_list

        self.logical_lengths = []
        self.physical_lengths = []
        self._manager = multiprocessing.Manager()
        self._read_counts = []
        self._rc_lock = multiprocessing.Lock()

        for cfg in self.config_list:
            inst = dataset.build_dataset(cfg.dataset, self.spec, self.hparams, cfg.kwargs)
            assert isinstance(inst, RandomSafeDataset), "Don't nest!"
            self.datasets.append(inst)
            inst_length = len(inst)
            self.physical_lengths.append(inst_length)
            self.logical_lengths.append(int(inst_length * cfg.get('subsample', 1.0)))
            self._read_counts.append(self._manager.dict())
            exp.logger.info(f"CombinedDataset component {cfg.dataset}, Logical = {self.logical_lengths[-1]}, "
                            f"Physical = {self.physical_lengths[-1]}")

        self.logical_cum_lengths = np.cumsum(self.logical_lengths)

    def __len__(self):
        return self.logical_cum_lengths[-1]

    def get_name(self):
        return "+".join([t.get_name() for t in self.datasets])

    def get_short_name(self):
        return "+".join([t.get_short_name() for t in self.datasets])

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.logical_cum_lengths, idx, side='right').item()
        item_idx = idx - (self.logical_cum_lengths[dataset_idx - 1].item() if dataset_idx > 0 else 0)

        with self._rc_lock:
            if item_idx not in self._read_counts[dataset_idx]:
                self._read_counts[dataset_idx][item_idx] = 0
            read_count = self._read_counts[dataset_idx][item_idx]
            physical_idx = (read_count * self.logical_lengths[dataset_idx] + item_idx) % \
                           self.physical_lengths[dataset_idx]
            self._read_counts[dataset_idx][item_idx] += 1

        res = self.datasets[dataset_idx][physical_idx]
        res[DS.DATASET_CFG] = self.config_list[dataset_idx]
        return res
