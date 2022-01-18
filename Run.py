# from DataPreprocessing import DataPreprocessing
from linear_genetic_programming.lgp_classifier import LGPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import metrics
import numpy as np
import time
start_time = time.time()
seed = np.random.randint(1000000)


# Dr. Steffany DLB data CTRL: 1, DLM: 0
df = pd.read_csv('dataset/sub_DLB_metabolomics.csv')
names = df.columns[5:]
y = df['Diagnosis'].to_numpy()
y = np.where(y == 'DLB', 0, y)
y = np.where(y == 'CTRL', 1, y)
y = y.tolist()
X = df.iloc[:, 5:].to_numpy()
scaler=MinMaxScaler((-1,1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)



# ---------------------------- Run -----------------------------------------------------------------
lgp = LGPClassifier(numberOfInput = X_train.shape[1], numberOfVariable = 30, populationSize = 400,
                            fitnessThreshold = 1.0, max_prog_ini_length = 80, min_prog_ini_length = 40,
                            maxGeneration = 70, tournamentSize = 8, showGenerationStat=True,
                            isRandomSampling=True, maxProgLength = 800, randomState=seed)
lgp.fit(X_train, y_train)
y_pred = lgp.predict(X_test)
y_prob = lgp.predict_proba(X_test)[:, 0]
lgp.testingAccuracy = accuracy_score(y_pred, y_test)
# calculate F1, AUC scores
f1_scores = metrics.f1_score(y_test, y_pred, pos_label=0)
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob, pos_label=0)
auc_scores = metrics.auc(fpr, tpr)
# store F1, AUC in validationScores
lgp.validationScores = {'f1':f1_scores, 'auc':auc_scores, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'seed': seed}

print(accuracy_score(y_pred, y_test))
print(lgp.bestEffProgStr_)
print(f1_scores)
print(auc_scores)
print("%s hours" % ((time.time() - start_time)/60/60))
lgp.save_model()

# y_pred = lgp.predict(X_test)
#
# result = [lgp.bestProFitness_, lgp.populationAvg_, round(accuracy_score(y_test, y_pred), 2), [lgp.bestEffProgStr_]]

# Best prog fitness,Final population Avg,Testing set accuracy,Effective Prog Str
# with open('lgp_result.txt', 'a') as f:
#     f.write('\t'.join(map(str, result)) + '\n')

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
#
# #%%
#
# draw ROC curve
# from sklearn.metrics import plot_roc_curve
# svc_disp = plot_roc_curve(lgp, X_test, y_test)
# s = str(os.getpid()) + 'ROC.png'
# plt.savefig(s)