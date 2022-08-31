import sklearn.datasets
import pandas as pd
import numpy as np

def gen_pub(safra):

    dset = sklearn.datasets.load_breast_cancer()
    X = dset.data

    return pd.DataFrame(range(0,X.shape[0]),columns=["user_id"])
