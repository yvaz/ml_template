from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

"""
Classe utilizada para compor métodos estáticos de métricas de avaliação de ML
@author Yule Vaz
"""

class MetricsCollection():
       
    """
    Método que calcula o melhor F1 dado os cortes dentro do predict_proba
    @param y_test Labels do conjunto de teste
    @param y_pred Retorno do predict_proba
    @param n_cuts Cortes ao longo do predict_proba
    @param log_scale Define se os cortes serão efetuados em escala logarítmica
    @return float Melhor score F1 ao longo do predict_proba
    """
    @staticmethod
    def best_f1(y_test,y_pred,n_cuts=100,log_scale=True):

        scores = []

        if log_scale:
            # Corte em escala logarítmica
            cuts = np.logspace(np.log10(min(y_pred)+1e-5),np.log10(max(y_pred)+1e-5),n_cuts)
        else:
            # Corte em escala uniforme
            cuts = np.arange(min(y_pred),max(y_pred),(max(y_pred)-min(y_pred))/n_cuts)

        y_cuts = []

        #Threshold
        for i in cuts:
            y_cut = np.copy(y_pred)
            y_cut[y_pred < i] = 0
            y_cut[y_pred >= i] = 1
            y_cuts.append(y_cut)

        #Aplica f1_score
        f1_batch = lambda x : f1_score(y_test,x)
        scores = list(map(f1_batch,y_cuts))

        return max(scores)