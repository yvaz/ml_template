import sklearn.datasets
import pandas as pd
import numpy as np

def gen_target(safra):

    dset = sklearn.datasets.load_breast_cancer()
    y = pd.DataFrame(dset.target, columns=["target"])
    user_id = pd.DataFrame(range(len(y)),columns=["user_id"])

    target = pd.concat([user_id,y],axis=1)

    return target.loc[target["target"] == 1]
