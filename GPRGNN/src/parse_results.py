import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--result_path', type=str, required=True,
                        help="Path to results txt.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    best_mean_score = 0
    best_std_score = 0
    cur_lr, cur_wd = None, None
    best_lr, best_wd = None, None
  
    with open(args.result_path, 'r') as f:
        for line in f:
            if '+-' in line:
                split_line = line.split(' ')[-3:]
                mean, std = float(split_line[0]), float(split_line[2])
                if mean > best_mean_score:
                    best_mean_score = mean
                    best_std_score = std   
    
    print(f'{best_mean_score:.2f} +- {best_std_score:.2f}')      
