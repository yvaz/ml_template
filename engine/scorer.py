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
from io_ml import io_metadata
from engine.main_cfg import MainCFG

class Scorer():

    def __init__(self,
                 safra: int,
                 config: str = os.path.dirname(__file__)+"/main_cfg.yaml"):
            
        self.main_cfg = MainCFG(config)
        self.logger = Logger(self)
        self.metadata = io_metadata.IOMetadata()
        self.logger.log('Inicializando processo de escoragem')
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
        
        if self.main_cfg.prod: 
            self.logger.log('Carregando pickle de S3')
            self.main_cfg.prod['params']['remote_path'] += self.main_cfg.model_name+'/registries/'
            self.main_cfg.prod['params']['local_path'] = './registries_tmp'
            self.prod_obj = self.main_cfg.prod_mod(**self.main_cfg.prod['params'])
            self.prod_obj.read_folder()
            os.system('rm -rf ./registries/pkl_*')
            os.system('mv ./registries_tmp/pkl_* ./registries/')
            os.system('rm -rf ./registries_tmp')
        
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
                                 
        io = self.main_cfg.config_mod(self.main_cfg.persist_method_score)
            
        res['SAFRA'] = datetime.strptime(du.DateUtils.add(self.safra,1),'%Y%m')
        res['MODEL_NAME'] = self.main_cfg.model_name
        res['DT_EXEC'] = self.metadata.metadata['executor']['score_timestamp']
        res['CUS_CUST_ID'] = res['CUS_CUST_ID'].astype(pd.Int64Dtype())
        
        self.logger.log('Score types:\n{}'.format(res.dtypes))      
        nulos = res.isnull().sum()
        self.logger.log('Nulos:\n{}'.format(nulos))
        
        io_c = io(**self.main_cfg.persist_params_score)
        io_c.write(res[['MODEL_NAME','CUS_CUST_ID','DECIL','SCORES_0','SCORES_1','SAFRA','DT_EXEC']])

