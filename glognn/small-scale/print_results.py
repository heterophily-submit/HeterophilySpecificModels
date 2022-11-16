import json
import numpy as np

dataset = 'workers'

for order in (2,):
    print(f'Order: {order}')
    for lr in (0.001, 0.003, 0.01, 0.03, 0.1):
        print(f'Learning rate: {lr}')
        for wd in (0.01, 0.001, 0.0001, '1e-05'):
            print(f'Weight decay: {wd}')
            test_accs = []
            for i in range(10):
                with open(f'runs/{dataset}_lr{lr}_do0.0_es100_wd{wd}_alpha0.0_beta1.0_gamma0.0_delta1.0_nlid1_nl2_ordersid2_orders{order}_split{i}_results.txt') as f:
                    data = json.load(f)
                test_accs.append(100 * data['test_acc'])

            print(f'{np.mean(test_accs):.2f} +- {np.std(test_accs):.2f}')
