import numpy as np

result_path = <path_to_results>

with open(result_path, 'r') as f:
    values = [float(line) for line in f]

print(f'{100 * np.mean(values):.2f} +- {100 * np.std(values):.2f}')