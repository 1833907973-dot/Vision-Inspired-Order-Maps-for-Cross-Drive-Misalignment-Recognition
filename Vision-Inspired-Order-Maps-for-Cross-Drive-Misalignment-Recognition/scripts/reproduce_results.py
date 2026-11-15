"""Recompute reported means/SDs from released per-seed result tables."""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'{label}: {a} differs from {b}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/results')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/summary')
    args = parser.parse_args()
    summaries = read_csv(args.data / 'extended_metric_summary.csv')
    seeds = read_csv(args.data / 'oof_seed_metrics.csv')
    groups = defaultdict(list)
    for row in seeds:
        groups[(row['experiment'], row['task'])].append(row)
    expected = {(r['experiment'], r['task']) for r in summaries}
    if set(groups) != expected or len(expected) != 27:
        raise ValueError('Expected matching nine-method by three-endpoint tables.')
    recomputed = []
    by_key = {}
    for row in summaries:
        key = (row['experiment'], row['task'])
        seed_rows = groups[key]
        if len(seed_rows) != int(row['n_model_seeds']) or len({r['seed'] for r in seed_rows}) != len(seed_rows):
            raise ValueError(f'Invalid seed grouping: {key}')
        result = {'experiment': key[0], 'task': key[1], 'n_seeds': len(seed_rows)}
        for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']:
            values = [float(r[metric]) for r in seed_rows]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            close(mean, row[metric + '_mean'], f'{key} {metric} mean')
            close(sd, row[metric + '_std'], f'{key} {metric} SD')
            result[metric + '_mean'] = mean
            result[metric + '_std'] = sd
        recomputed.append(result)
        by_key[key] = result
    confusion = read_csv(args.data / 'epcda_five_class_confusion_summary.csv')
    if len(confusion) != 25:
        raise ValueError('Expected a 5 by 5 confusion table.')
    diagonal = []
    for i in range(5):
        cells = [r for r in confusion if int(r['true_class_index']) == i]
        close(sum(float(r['row_normalized_rate_mean_across_model_seeds']) for r in cells), 1.0, 'confusion row sum')
        diagonal.extend(float(r['row_normalized_rate_mean_across_model_seeds']) for r in cells if int(r['predicted_class_index']) == i)
    close(statistics.mean(diagonal), by_key[('epcda', 'five_class')]['balanced_accuracy_mean'], 'confusion-derived BAcc')
    target = json.loads((args.data / 'target_supervised_summary.json').read_text(encoding='utf-8'))
    for metric, summary in target['metrics'].items():
        values = [r[metric] for r in target['seed_oof']]
        close(statistics.mean(values), summary['mean'], 'target supervised mean')
        close(statistics.stdev(values), summary['sd'], 'target supervised SD')
    paired = read_csv(args.data / 'extended_matched_paired_tests_24.csv')
    if len(paired) != 24:
        raise ValueError('Expected 24 matched contrasts.')
    for row in paired:
        task = row['task']
        delta = by_key[('epcda', task)]['balanced_accuracy_mean'] - by_key[(row['baseline'], task)]['balanced_accuracy_mean']
        close(delta, row['epcda_minus_baseline_bacc'], 'paired mean difference')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'recomputed_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(recomputed[0]))
        writer.writeheader()
        writer.writerows(recomputed)
    report = {
        'status': 'passed',
        'method_endpoint_groups_checked': len(recomputed),
        'paired_mean_differences_checked': len(paired),
        'epcda_five_class_bacc_percent': 100 * by_key[('epcda', 'five_class')]['balanced_accuracy_mean'],
        'target_supervised_five_class_bacc_percent': 100 * target['metrics']['five_class_bacc']['mean'],
        'intervals_and_p_values': 'Stored original outputs are supplied; they are not re-estimated from the small signal sample.',
        'interpretation': 'The target-supervised reference is a separate protocol, not a direct performance competitor for unsupervised adaptation.',
    }
    (args.output / 'result_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
