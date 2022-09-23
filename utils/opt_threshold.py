from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import roc_curve
from collections_ml import plots_collection as pc
import matplotlib.pyplot as plt
import numpy as np

class OptThreshold(BaseEstimator, TransformerMixin):
    
    def __init__(self,clf):
        
        self.clf = clf
        
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
    
    def transform(self,X,y=None):
        
        proba = self.clf.predict_proba(X)[:,1]
        proba[proba >= self.threshold] = 1
        proba[proba < self.threshold] = 0
        return proba
       
    def predict(self,X,y=None):
        
        proba = self.clf.predict_proba(X)[:,1]
        proba[proba >= self.threshold] = 1
        proba[proba < self.threshold] = 0
        return proba
    
           
    def predict_proba(self,X,y=None):
        
        proba = self.clf.predict_proba(X)
        return proba
        
            
        
        
    
    