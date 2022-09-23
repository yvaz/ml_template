from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class MetricsCollection():
       
    @staticmethod
    def best_f1(y_test,y_pred,n_cuts=100,log_scale=True):

        scores = []

        if log_scale:
            cuts = np.logspace(np.log10(min(y_pred)+1e-5),np.log10(max(y_pred)+1e-5),n_cuts)
        else:
            cuts = np.arange(min(y_pred),max(y_pred),(max(y_pred)-min(y_pred))/n_cuts)

        y_cuts = []

        for i in cuts:
            y_cut = np.copy(y_pred)
            y_cut[y_pred < i] = 0
            y_cut[y_pred >= i] = 1
            y_cuts.append(y_cut)

        f1_batch = lambda x : f1_score(y_test,x)
        scores = list(map(f1_batch,y_cuts))

        return max(scores)