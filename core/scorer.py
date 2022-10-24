import yaml
from yaml.loader import SafeLoader
import pickle
import pandas as pd
import numpy as np
import importlib
import os
import re
from utils import date_utils as du
from utils.logger import Logger
from datetime import datetime
from io_ml.io_parquet import IOParquet

class Scorer():

    def __init__(self,
                 safra: int,
                 config: str = "core/main_cfg.yaml"):

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
            
        self.logger = Logger(self)
        self.logger.log('Inicializando processo de escoragem')
        
        self.model_name = self.config['model_name']
        self.persist_package = self.config['score']['persist_method']['package']
        self.persist_module = self.config['score']['persist_method']['module']
        
        self.persist_params = {}
        
        if 'params' in self.config['score']['persist_method'].keys():
            self.persist_params = self.config['score']['persist_method']['params']
            
        self.persist_params['tb_name'] = self.config['score']['tb_name']
        self.model_name = self.config['model_name']
        self.safra = safra
        
    def get_last_trained_safra(self):
        
        path = 'registries/'
        
        pkls = sorted(
                filter(
                lambda x: bool(re.search(r"pkl_[0-9]+", x)),
                os.listdir(path)
            )
        )
        
        safras = np.array(
                     list(
                         map(
                            lambda x : int(x.replace('pkl_','')),
                            pkls
                        )
                    )
                 )
        
        safras = safras[safras <= int(self.safra)]
        
        return max(safras)
        
        

    def load_model(self):
        
        self.logger.log('Carregando modelo para escoragem')
        
        safra_alvo = self.get_last_trained_safra()
        
        path = 'registries/pkl_{safra}/'.format(safra=safra_alvo)
        
        list_of_files = sorted(
                            filter(
                                lambda x: os.path.isfile(os.path.join(path, x)),
                                os.listdir(path)
                            )
                        )
        
        last_pickle = list_of_files[-1]
        
        self.logger.log('Pickle utilizado: {path}{last_pickle}'.format(path=path,last_pickle=last_pickle))
        
        with open(path+last_pickle,'rb') as fp:
            self.model = pickle.load(fp)

    def score(self,X,y=None):
        
        self.logger.log('Produzindo scores')
        
        _,result = self.model.transform(X.drop('CUS_CUST_ID',axis=1))
        
        res = pd.DataFrame(np.column_stack([X['CUS_CUST_ID'].to_numpy(),result]),
                       columns=['CUS_CUST_ID','PR_0', 'PR_1'])
        
        res['DECIL'] = (pd.qcut(res['PR_0'].rank(method='first'), 10, labels=False)+1).astype(pd.Int64Dtype())
        
        res['SCORES_0'] = (res['PR_0'].astype('Float64')*10000).astype('Int64')
        res['SCORES_1'] = (res['PR_1'].astype('Float64')*10000).astype('Int64')
                                 
        mod = importlib.import_module(self.persist_package)
        io = getattr(mod,self.persist_module)    
            
        res['SAFRA'] = datetime.strptime(du.DateUtils.add(self.safra,1),'%Y%m')
        res['MODEL_NAME'] = self.model_name
        res['DT_EXEC'] = datetime.now().strftime('%Y%m%d%H%M%S')
        res['CUS_CUST_ID'] = res['CUS_CUST_ID'].astype(pd.Int64Dtype())
        
        self.logger.log('Score types:\n{}'.format(res.dtypes))      
        nulos = res.isnull().sum()
        self.logger.log('Nulos:\n{}'.format(nulos))
        
        io_c = io(**self.persist_params)
        io_c.write(res[['MODEL_NAME','CUS_CUST_ID','DECIL','SCORES_0','SCORES_1','SAFRA','DT_EXEC']])

