import sklearn.datasets
import pandas as pd
import random
import numpy as np

def gen_features(safra):

    dset = sklearn.datasets.load_breast_cancer()
    X = pd.DataFrame(dset.data, columns = dset.feature_names)
    ids = pd.DataFrame(range(X.shape[0]), columns = ["user_id"])
    
    cats = np.random.choice(['a','b','c','d','d'],X.shape[0])
    cats_pd = pd.DataFrame(cats,columns=['var1'])
    feats = pd.concat([X,cats_pd],axis=1)
    feats = pd.concat([ids,feats],axis=1)

    return feats 
