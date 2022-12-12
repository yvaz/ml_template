from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import roc_curve
from collections_ml import plots_collection as pc
import matplotlib.pyplot as plt
import numpy as np

"""
Classe otimizadora de threshold para os modelos de ML
"""
class OptThreshold(BaseEstimator, TransformerMixin):
    
    """
    Construtor
    @param clf Modelo de ML produzido
    """
    def __init__(self,clf):
        
        self.clf = clf
        
    """
    Método que aplica o fit do pipe
    @param X Features
    @param y Labels
    @return self OptThreshold
    """
    def fit(self,X,y):
        preds = self.clf.predict_proba(X)[:,1]
        fpr, tpr, thresholds = roc_curve(y,preds)
        pc.PlotsCollection.roc_curve_plot(preds,y)
        plt.savefig('train_roc_curve.png')
        plt.close()
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        self.threshold = thresholds[ix]
        
        return self
    
    """
    Método que transforma os dados
    @param X Features
    @param y Labels
    @return Classes 0 e 1 aplicando o threshold ótimo
    """
    def transform(self,X,y=None):
        
        proba = self.clf.predict_proba(X)[:,1]
        proba[proba >= self.threshold] = 1
        proba[proba < self.threshold] = 0
        return proba
    
    """
    Método que transforma os dados
    @param X Features
    @param y Labels
    @return int Classes 0 e 1 aplicando o threshold ótimo
    """      
    def predict(self,X,y=None):
        
        proba = self.clf.predict_proba(X)[:,1]
        proba[proba >= self.threshold] = 1
        proba[proba < self.threshold] = 0
        return proba
        
    """
    Retorna predict_proba
    @param X Features
    @param y Labels
    @return float predict_proba
    """          
    def predict_proba(self,X,y=None):
        
        proba = self.clf.predict_proba(X)
        return proba
        
            
        
        
    
    