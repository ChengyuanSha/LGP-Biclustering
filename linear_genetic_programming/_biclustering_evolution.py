


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
from sklearn.cluster import KMeans


# load Dr. Steffany data
# 1: control, 0: case
path2pickle = '../dataset/lgp702SteffanyNew.pkl'
df = pd.read_csv('../dataset/Discovery_Cohort2020_08_25_forTing_imputed_fifth_min.csv')
df_disease = df[df['Diagnosis'] == 'DLB']
df_ctrl = df[df['Diagnosis'] == 'CTRL']
names = df.columns[5:]

y_disease = df_disease['Diagnosis'].to_numpy()
y_disease = np.where(y_disease == 'DLB', 0, y_disease)
y_disease = np.where(y_disease == 'CTRL', 1, y_disease)
# y_disease = y_disease.tolist()

y_ctrl = df_ctrl['Diagnosis'].to_numpy()
y_ctrl = np.where(y_ctrl == 'DLB', 0, y_ctrl)
y_ctrl = np.where(y_ctrl == 'CTRL', 1, y_ctrl)
y_ctrl = y_ctrl.tolist()

X_disease = df_disease.iloc[:, 5:].to_numpy()
X_ctrl = df_ctrl.iloc[:, 5:].to_numpy()
scaler = MinMaxScaler((-1, 1))
X_disease = scaler.fit_transform(X_disease)
X_ctrl = scaler.fit_transform(X_ctrl)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X_disease)

pop1 = X_disease[kmeans.labels_ == 0, :]
pop1_y = y_disease[kmeans.labels_ == 0]
pop2 = X_disease[kmeans.labels_ == 1, :]

pop1 = np.append(pop1, X_ctrl[0:pop1.shape[0], :], axis=0)

pop2 = np.append(pop2, X_ctrl[pop1.shape[0]:(pop1.shape[0]+pop2.shape[0]), :], axis=0)


