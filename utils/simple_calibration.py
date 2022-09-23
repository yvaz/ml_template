from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class SimpleCalibration(BaseEstimator, TransformerMixin):
    
    def __init__(self,clf,n_bins):
        
        self.clf = clf
        self.n_bins = n_bins
        
        
    def _get_cuts(self,X):
        
        preds = self.clf.predict_proba(X)
        max_preds = max(preds[:,1])
        min_preds = min(preds[:,1])
        cuts = pd.cut(preds[:,1],self.n_bins,labels=range(0,self.n_bins))
        
        print(cuts)
        
        pred_cuts = pd.DataFrame({'predict_0':preds[:,0],'predict_1':preds[:,1],'bins':cuts})
        return pred_cuts
        
        
    def fit(self,X,y):
        
        pred_cuts = self._get_cuts(X)
        
        y_pd = pd.DataFrame(y,columns=['target'])
        counts = pd.concat([pred_cuts,y_pd],axis=1)\
                                .groupby('bins').agg(probs=('target','sum'),cnt=('target','count'))\
                                .reset_index()
        
        counts['probs'] = (counts['probs']/counts['cnt'])
        
        self.probs = counts
        
        return self
    
    def transform(self,X,y=None):
        
        pred_cuts = self._get_cuts(X)
        
        ret = self.probs.merge(pred_cuts,'right',on='bins')['probs'].fillna(0)
        
        ret[ret >= 0.5] = 1
        ret[ret < 0.5] = 0
        
        return ret.to_array()
    
    def predict(self,X):
        
        pred_cuts = self._get_cuts(X)
        
        ret = self.probs.merge(pred_cuts,'right',on='bins')['probs'].fillna(0)
        
        ret[ret >= 0.5] = 1
        ret[ret < 0.5] = 0
        
        return ret.to_array()
    
    def predict_proba(self,X):
        
        pred_cuts = self._get_cuts(X)
        
        ret = self.probs.merge(pred_cuts,'right',on='bins')\
                    .fillna(0).drop(['bins','cnt'],axis=1)
        ret['predict_0'] = 1-ret['probs']
        ret['predict_1'] = ret['probs']
        ret = ret.drop('probs',axis=1)
        
        return ret.to_numpy()
        
        
        
        
        
        
        