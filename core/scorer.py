import yaml
from yaml.loader import SafeLoader
import pickle
import pandas as pd
import numpy as np
import importlib
import os
import re
from utils import date_utils as du

class Scorer():

    def __init__(self,
                 safra: int,
                 config: str = "core/main_cfg.yaml"):

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
            
        self.persist_package = self.config['score']['persist_method']['package']
        self.persist_module = self.config['score']['persist_method']['module']
        self.persist_params = self.config['score']['persist_method']['params']
        self.persist_params['tb_name'] = self.config['score']['tb_name']
        self.model_name = self.config['model_name']
        self.safra = safra
        
    def get_last_trained_safra(self):
        
        path = 'registries/'
        print(os.listdir(path))
        pkls = sorted(
                filter(
                lambda x: bool(re.search(r"pkl_[0-9]+", x)),
                os.listdir(path)
            )
        )
        
        print(pkls)
        safras = np.array(
                     list(
                         map(
                            lambda x : int(x.replace('pkl_','')),
                            pkls
                        )
                    )
                 )
        
        print(safras <= int(self.safra))
        safras = safras[safras <= int(self.safra)]
        
        return max(safras)
        
        

    def load_model(self):
        
        safra_alvo = self.get_last_trained_safra()
        
        path = 'registries/pkl_{safra}/'.format(safra=safra_alvo)
        
        list_of_files = sorted(
                            filter(
                                lambda x: os.path.isfile(os.path.join(path, x)),
                                os.listdir(path)
                            )
                        )
        
        last_pickle = list_of_files[-1]
        
        with open(path+last_pickle,'rb') as fp:
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
            
        res['safra'] = du.DateUtils.add(self.safra,1)
        
        io_c = io(**self.persist_params)
        io_c.write(res[['CUS_CUST_ID','scr_grp','scores_0','scores_1','safra']])

