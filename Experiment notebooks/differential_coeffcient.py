from linear_genetic_programming.lgp_classifier import LGPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from matplotlib.pyplot import figure
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto


def get_diff_coefficient():
    # Dr. Steffany data
    # 1: control, 0: case
    path2pickle = '../dataset/lgp_steffany_5_19.pkl'
    df = pd.read_csv('../dataset/DLB_sub_lgp.pkl')
    # names = df.columns[:-1]
    # y = df['category'].to_numpy()
    #
    # X = df.loc[:, df.columns != 'category'].to_numpy()
    # scaler=MinMaxScaler((-1,1))
    # X = scaler.fit_transform(X)

    data = df.loc[:, df.columns != 'category']

    # only keep 1
    data = data.loc[df['category'] == 1, :]
    corr = data.corr()

    corr_df_control = corr.stack().reset_index().rename(columns={'level_0':'Source','level_1':'Target', 0:'Weight'})

    data0 = df.loc[:, df.columns != 'category']
    data0 = data0.loc[df['category'] == 0, :]


    corr0 = data0.corr()
    corr_df0_case = corr0.stack().reset_index().rename(columns={'level_0':'Source','level_1':'Target', 0:'Weight'})


    result = corr_df0_case.copy()


    result['Weight'] = corr_df0_case['Weight'] - corr_df_control['Weight']

    return result, corr_df_control, corr_df0_case



if __name__ == '__main__':
    result, corr_df_control, corr_df0_case = get_diff_coefficient()
    print(corr_df_control)


    # sns.distplot(result['Weight'])
    # plt.show()

    import statistics
    print(statistics.stdev(result['Weight']))
    print(statistics.mean(result['Weight']))
