import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import seaborn as sns
import pandas as pd
import numpy as np

"""
Classe utilizada para compor métodos estáticos de plot
"""
class PlotsCollection():
    
    """
    Método que apresenta o gráfico de calibração (probabilidade real vs. probabilidade aferida pelo modelo)
    @param y_test Labels do conjunto de teste
    @param y_pred Retorno do predict_proba
    """
    @staticmethod
    def calibration_plot(y_pred,y_true):

        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        plt.plot(prob_true,prob_pred)

    """
    Método que apresenta a curva ROC
    @param preds Labels do conjunto de teste
    @param y Retorno do predict_proba
    """
    @staticmethod
    def roc_curve_plot(preds,y):

        fpr, tpr, threshold = roc_curve(y, preds)
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    """
    Histograma do predict_proba vs classe 1 e classe 0
    @param preds Labels do conjunto de teste
    @param y Retorno do predict_proba
    """
    @staticmethod
    def targets_plot(preds,y):
        
        df = pd.DataFrame({'scores':preds,'targets':y})
        sns.histplot(data=df, x='scores', hue='targets', stat="density", common_norm=False)

    """
    Gráfico que apresenta Y real acumulado e Y previsto acumulado
    @param preds Labels do conjunto de teste
    @param y Retorno do predict_proba
    """       
    @staticmethod
    def reg_sort_plot(preds,y):
              
        idx = np.argsort(y)
        
        plt.plot(np.log(np.array(y)[idx]+1e-5))
        plt.plot(np.log(np.array(preds)[idx]+1e-5))