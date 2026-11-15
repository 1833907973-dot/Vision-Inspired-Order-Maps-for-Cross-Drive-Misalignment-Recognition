"""Exercise the compact EPCDA model on real sample data; no trained weights."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data_utils import load_sample, split_indices, Normalizer, make_batch
from epcda_model import EPCDAModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/sample/real_signal_sample.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/demo')
    parser.add_argument('--fold', type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.set_num_threads(2)
    data = load_sample(args.data)
    splits = split_indices(data, args.fold)
    normalizer = Normalizer.fit_source_train(data['X'][splits['source_train']])
    source = make_batch(data, splits['source_train'][:8], normalizer)
    target = make_batch(data, splits['target_adapt'][:8], normalizer, mask_labels=True)
    if not (target['y'] == -1).all():
        raise RuntimeError('Target adaptation labels must be masked.')
    x = torch.from_numpy(np.concatenate([source['x'], target['x']]))
    rpm = torch.from_numpy(np.concatenate([source['rpm_raw'], target['rpm_raw']]))
    fs = torch.from_numpy(np.concatenate([source['fs'], target['fs']]))
    model = EPCDAModel().eval()
    with torch.no_grad():
        output = model(x, rpm, fs)
    for key in ['cls', 'front_features', 'amplitude', 'coherence']:
        if not torch.isfinite(output[key]).all():
            raise RuntimeError(f'Non-finite output: {key}')
    deployment_modules = [model.front_end, model.encoder, model.fault_projector, model.classifier]
    deployment_count = sum(p.numel() for module in deployment_modules for p in module.parameters())
    if deployment_count != 77077:
        raise RuntimeError(f'Unexpected model size: {deployment_count}')
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'demo_features.npz',
                        features=output['front_features'].numpy(),
                        amplitude=output['amplitude'].numpy(),
                        coherence_real=output['coherence'].real.numpy(),
                        coherence_imag=output['coherence'].imag.numpy(),
                        untrained_logits=output['cls'].numpy())
    report = {
        'status': 'passed',
        'weights': 'random initialization; outputs are a software demonstration, not diagnostic predictions',
        'normalization': 'statistics fitted only to source_train examples in the released subset',
        'split_sizes': {k: len(v) for k, v in splits.items()},
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'deployment_parameters': deployment_count,
        'input_shape': list(x.shape),
        'feature_shape': list(output['front_features'].shape),
        'target_adaptation_labels_masked': True,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
    }
    (args.output / 'demo_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
