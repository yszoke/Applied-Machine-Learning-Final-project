import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import numpy as np

def friedman_test(adaboost_path, DIB_path, DIB2_path):
    adaboost_data = pd.read_csv(adaboost_path)
    DIB_data = pd.read_csv(DIB_path)
    DIB2_data = pd.read_csv(DIB2_path)
    stat, p = friedmanchisquare(adaboost_data['AUC'], DIB_data['AUC'], DIB2_data['AUC'])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
        posthocs_test(adaboost_data['AUC'], DIB_data['AUC'], DIB2_data['AUC'])

def posthocs_test(adaboost_data, DIB_data, DIB2_data):
    data = np.array([adaboost_data, DIB_data, DIB2_data])

    # perform Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data.T))

if __name__ == '__main__':
    # for each algorithm and dataset->do friedman test:
    friedman_test('results\\result_Adaboost.csv',
                  'results\\result_DIB.csv',
                  'results\\result_DIB2.csv')