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

    all_results = []
    with open(args.result_path, 'r') as f:
        for line in f:
            if line.startswith('best valf'):
                res = float(line.split(' ')[-1])
                all_results.append(res)
    
    print(f'{100 * np.mean(all_results):.2f} +- {100 * np.std(all_results):.2f}')
