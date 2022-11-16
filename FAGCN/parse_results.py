import argparse
import numpy as np

from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--result_path', type=str, required=True,
                        help="Path to results txt.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.result_path, 'r') as f:
        current_split = 0
        best_results_for_split = defaultdict(float)
        for line in f:
            if line.startswith('Split'):
                split_id = int(line.split(' ')[-1])
            elif line.startswith('0.'):
                metric_value = float(line)
                if best_results_for_split[split_id] < metric_value:
                    best_results_for_split[split_id] = metric_value

        metric_values = [100 * v for k, v in best_results_for_split.items()]
        print(f'Averaged results: {np.mean(metric_values):.2f} +- {np.std(metric_values):.2f}')
            