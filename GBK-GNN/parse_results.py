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

    current_split = 0
    best_res_for_split = defaultdict(float)
    with open(args.result_path, 'r') as f:
        for line in f:
            if line.startswith('load'):
                current_split = int(line.split(':')[-1].lstrip(' '))
            elif line.startswith('Final'):
                res = float(line.split(':')[-1])
                if res > best_res_for_split[current_split]:
                    best_res_for_split[current_split] = res
    
    all_results = [v for _, v in best_res_for_split.items()]
    print(f'{100 * np.mean(all_results):.2f} +- {100 * np.std(all_results):.2f}')
