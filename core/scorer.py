import yaml
from yaml.loader import SafeLoader
import pickle
import pandas as pd
import numpy as np
import importlib

class Scorer():

    def __init__(self,
                 config: str = "core/main_cfg.yaml"):

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
            
        self.persist_package = self.config['score']['persist_method']['package']
        self.persist_module = self.config['score']['persist_method']['module']
        self.persist_params = self.config['score']['persist_method']['params']
        self.persist_params['tb_name'] = self.config['score']['tb_name']
        self.model_name = self.config['model_name']
        

    def load_model(self):

        with open('registries/'+self.model_name+'.pkl','rb') as fp:
            self.model = pickle.load(fp)

    def score(self,X,y=None):
        
        print(X.columns)
        _,result = self.model.transform(X.drop('CUS_CUST_ID',axis=1))
        
        res = pd.DataFrame(np.column_stack([X['CUS_CUST_ID'].to_numpy(),result]),
                       columns=['CUS_CUST_ID','PR_0', 'PR_1'])
        res['scr_grp'] = pd.qcut(res['PR_0'].rank(method='first'), 10, labels=False)+1
        res['scores_0'] = res['PR_0']
        res['scores_1'] = res['PR_1']
                                 
        mod = importlib.import_module(self.persist_package)
        io = getattr(mod,self.persist_module)    
            
        io_c = io(**self.persist_params)
        io_c.write(res[['CUS_CUST_ID','scores_0','scores_1']])

