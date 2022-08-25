import sklearn.datasets
import pandas as pd

def gen_feats(safra):

    dset = sklearn.datasets.load_iris()
    X = dset.data
    y = dset.target

    X = X[10:,:]
    y = y[10:]

    return pd.DataFrame(X, columns=dset.feature_names), pd.DataFrame(y,columns=['target'])
