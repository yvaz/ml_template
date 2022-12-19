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
from io_ml.io_bq import IO_BQ
from io_ml import io_metadata
from engine.main_cfg import MainCFG
from decimal import Decimal

"""
Classe que efetua a escoragem do modelo
"""
class Scorer():

    """
    Construtor
    @param safra Safra de escoragem
    @param config Arquivo de configuração principal
    """
    def __init__(self,
                 safra: int,
                 config: str = os.path.dirname(__file__)+"/main_cfg.yaml"):
            
        self.main_cfg = MainCFG(config)
        self.logger = Logger(self)
        self.metadata = io_metadata.IOMetadata()
        self.logger.log('Inicializando processo de escoragem')
        self.safra = safra
        
    """
    Retorna a última safra treinada
    @return str Última safra treinada
    """
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

    """
    Seleciona o champing challenger da tabela definida no arquivo de configuração principal
    @return str A versão do melhor modelo
    """
    def _load_champ_challenger(self):
        
        io_bq = IO_BQ(self.main_cfg.persist_params_champ['tb_name'])
        
        query = """SELECT SAFRA,DT_TRAIN,METRIC BEST_METRIC FROM 
                   {tb_name}
                   WHERE SAFRA <= PARSE_DATE('%Y%m','{safra}')
                   AND MODEL_NAME='{model_name}'
                   ORDER BY METRIC DESC"""\
                .format(tb_name=self.main_cfg.persist_params_champ['tb_name'],
                        safra=self.safra,
                        model_name=self.main_cfg.model_name)
        
        if io_bq.read(query).shape[0] > 0:
            best_model = io_bq.read(query).loc[0]
            return best_model['SAFRA'].strftime('%Y%m'),best_model['DT_TRAIN']
        else:
            best_model = None
            return None,best_model 
    
    """
    Baixa o modelo do S3
    """    
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
        
        self.train_safra,self.train_dt = self._load_champ_challenger()
        if not self.main_cfg.champ_challenger or self.train_dt is None:
            self.train_safra = self.get_last_trained_safra()

            path = 'registries/pkl_{safra}/'.format(safra=self.train_safra)

            list_of_files = sorted(
                                filter(
                                    lambda x: os.path.isfile(os.path.join(path, x)),
                                    os.listdir(path)
                                )
                            )

            last_pickle = list_of_files[-1]
            pickle_suf = re.findall(r'[0-9]+\.pkl',last_pickle)[0]
            self.train_dt = pickle_suf.replace('.pkl','')
        else:
            path = 'registries/pkl_{safra}/'.format(safra=self.train_safra)
            last_pickle = self.main_cfg.model_name+'_'+str(self.train_dt)+'.pkl'
            
        self.logger.log('Train date: {train_dt}'.format(train_dt=self.train_dt))
        self.logger.log('Pickle utilizado: {path}{last_pickle}'.format(path=path,last_pickle=last_pickle))
        
        with open(path+last_pickle,'rb') as fp:
            self.model = pickle.load(fp)
           
    """
    Efetua o score de classificação e escreve 
    no local determinado no arquivo de config. principal
    """
    def score_class(self,X,y):
            _,result = self.model.transform(X.drop('CUS_CUST_ID',axis=1))        
            res = pd.DataFrame(np.column_stack([X['CUS_CUST_ID'].to_numpy(),result]),
                           columns=['CUS_CUST_ID','PR_0', 'PR_1'])

            res['DECIL'] = (pd.qcut(res['PR_0'].rank(method='first'), 10, labels=False)+1).astype(pd.Int64Dtype())

            res['SCORES_0'] = (res['PR_0'].astype('Float64')*10000).astype('Int64')
            res['SCORES_1'] = (res['PR_1'].astype('Float64')*10000).astype('Int64')

            io = self.main_cfg.config_mod(self.main_cfg.persist_method_score)

            res['SAFRA'] = datetime.strptime(du.DateUtils.add(self.safra,1),'%Y%m')
            res['MODEL_NAME'] = self.main_cfg.model_name
            res['DT_EXEC'] = datetime.now().strftime('%Y%m%d%H%M%S')
            res['DT_TRAIN'] = self.train_dt
            res['CUS_CUST_ID'] = res['CUS_CUST_ID'].astype(pd.Int64Dtype())

            self.logger.log('Score types:\n{}'.format(res.dtypes))      
            nulos = res.isnull().sum()
            self.logger.log('Nulos:\n{}'.format(nulos))

            io_c = io(**self.main_cfg.persist_params_score)
            io_c.write(res[['MODEL_NAME','CUS_CUST_ID','DECIL','SCORES_0','SCORES_1','SAFRA','DT_EXEC','DT_TRAIN']])  
            
    """
    Efetua o score de classificação e escreve 
    no local determinado no arquivo de config. principal
    """           
    def score_regr(self,X,y):
            result = self.model.transform(X.drop('CUS_CUST_ID',axis=1))        
            res = pd.DataFrame(np.column_stack([X['CUS_CUST_ID'].to_numpy(),result]),
                           columns=['CUS_CUST_ID','PR_0'])

            res['DECIL'] = (pd.cut(res['PR_0'].rank(method='first'), 
                                    self.main_cfg.bins, 
                                    labels=False)+1).astype(pd.Int64Dtype())  

            cast_dec = lambda x : Decimal('{:.2f}'.format(x))
            res['SCORES'] = list(map(cast_dec,res['PR_0']))
            
            self.logger.log('BINS: \n{bins}'.format(bins=self.main_cfg.bins))
            self.logger.log('SCORES: \n{score}'.format(score=res['SCORES']))

            io = self.main_cfg.config_mod(self.main_cfg.persist_method_score)

            res['SAFRA'] = datetime.strptime(du.DateUtils.add(self.safra,1),'%Y%m')
            res['MODEL_NAME'] = self.main_cfg.model_name
            res['DT_EXEC'] = datetime.now().strftime('%Y%m%d%H%M%S')
            res['DT_TRAIN'] = self.train_dt
            res['CUS_CUST_ID'] = res['CUS_CUST_ID'].astype(pd.Int64Dtype())

            self.logger.log('Score types:\n{}'.format(res.dtypes))      
            nulos = res.isnull().sum()
            self.logger.log('Nulos:\n{}'.format(nulos))

            io_c = io(**self.main_cfg.persist_params_score)
            io_c.write(res[['MODEL_NAME','CUS_CUST_ID','DECIL','SCORES','SAFRA','DT_EXEC','DT_TRAIN']])  
    
    """
    Efetua o score (classifc., regress.)
    @param X Features
    @param y Labels
    """
    def score(self,X,y=None):
        
        self.logger.log('Produzindo scores')
        
        if self.main_cfg.type == 'classification':
            self.score_class(X,y)
            
        elif self.main_cfg.type == 'regression':
            self.score_regr(X,y)

