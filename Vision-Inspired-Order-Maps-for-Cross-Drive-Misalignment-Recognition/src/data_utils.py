"""Portable data utilities adapted from the study's training code."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def load_sample(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    n = len(data['X'])
    if data['X'].shape != (n, 3, 128) or data['rpm_raw'].shape != (n, 128):
        raise ValueError('Expected X [N,3,128] and rpm_raw [N,128].')
    if data['channel_names'].tolist() != ['rpm', 'iq', 'id']:
        raise ValueError('Unexpected channel order.')
    for field in ('X', 'rpm_raw', 'fs_hz'):
        if not np.isfinite(data[field]).all():
            raise ValueError(f'{field} contains non-finite values.')
    if not (data['fs_hz'] > 0).all():
        raise ValueError('Sampling rates must be positive.')
    return data


def split_indices(data, fold=0):
    """Apply the original role rules to the released subset."""
    if fold not in (0, 1, 2):
        raise ValueError('fold must be 0, 1, or 2')
    p, trial = data['power'], fold + 1
    target = p == 1500
    masks = {
        'source_train': (p == 200) | ((p == 400) & (data['trial'] != trial)),
        'source_val': (p == 400) & (data['trial'] == trial),
        'target_adapt': target & (data['fold'] == fold),
        'target_test': target & (data['fold'] == (fold + 1) % 3),
        'target_probe': target & (data['fold'] == (fold + 2) % 3),
    }
    splits = {k: np.flatnonzero(v) for k, v in masks.items()}
    if any(len(v) == 0 for v in splits.values()):
        raise ValueError('An empty role prevents this demo from running.')
    run_sets = {k: set(data['run_id'][v].tolist()) for k, v in splits.items()}
    for left, right in [('source_train', 'source_val'), ('target_adapt', 'target_test'), ('target_adapt', 'target_probe'), ('target_test', 'target_probe')]:
        if run_sets[left] & run_sets[right]:
            raise ValueError(f'Run overlap: {left}, {right}')
    return splits


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit_source_train(cls, x):
        mean = x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = x.std(axis=(0, 2), keepdims=True).astype(np.float32)
        return cls(mean, np.maximum(std, 1e-6))

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32, copy=False)


def make_batch(data, indices, normalizer, mask_labels=False):
    """Keep physical rpm unscaled; mask target labels when requested."""
    return {
        'x': normalizer.transform(data['X'][indices]),
        'rpm_raw': data['rpm_raw'][indices],
        'fs': data['fs_hz'][indices],
        'y': np.full(len(indices), -1, dtype=np.int64) if mask_labels else data['y'][indices].copy(),
    }
