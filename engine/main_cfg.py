from utils.singleton import Singleton
import yaml
from yaml.loader import SafeLoader
import importlib
import imp
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from io_ml.io_parquet import IOParquet
import os
from os.path import exists
from utils import date_utils as du
from io_ml import io_metadata
from utils.logger import Logger

"""
Classe que armazena informação do arquivo de configuração principal 
"""
class MainCFG(Singleton):
    
    """
    Construtor
    @param config Arquivo de configuração principal
    """
    def __init__(self,
                 config: str = os.path.dirname(__file__)+'/main_cfg.yaml'):
        
        self.logger = Logger(self)
        self.logger.log('Configurando main_cfg singleton.')
        
        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
            
        self.model_name = self.config['model_name']
        self.type = self.config['type']
        self.train_test_sample = self.config['train_test_sample']
        self.persist = self.config['persist']  
        
        #CONFIGURA PERSISTÊNCIA DA ESCORAGEM
        
        self.persist_method_score = self.config['score']['persist_method']
        self.persist_package_score = self.config['score']['persist_method']['package']
        self.persist_module_score = self.config['score']['persist_method']['module']
        if 'params' in self.config['score']['persist_method'].keys():
            self.persist_params_score = self.config['score']['persist_method']['params']
        else:
            self.persist_params_score = {}
            
        self.persist_params_score['tb_name'] = self.config['score']['tb_name']
        
        #CONFIGURA PERSISTÊNCIA DA AVALIAÇÃO
        
        self.persist_method_eval = self.config['eval']['persist_method']
        self.persist_package_eval = self.config['eval']['persist_method']['package']
        self.persist_module_eval = self.config['eval']['persist_method']['module']
        
        if 'params' in self.config['eval']['persist_method'].keys():
            self.persist_params_eval = self.config['eval']['persist_method']['params']
        else:
            self.persist_params_eval = {}
            
        self.persist_params_eval['tb_name'] = self.config['eval']['tb_name']
        
        #CONFIGURA PERSISTÊNCIA DO CHAMPION CHALLENGER
        
        self.champ_challenger = None
        
        if 'champ_challenger' in self.config.keys():
        
            self.persist_method_champ = self.config['champ_challenger']['persist_method']
            self.champ_challenger = self.config['champ_challenger']
            self.persist_package_champ = self.config['champ_challenger']['persist_method']['package']
            self.persist_module_champ = self.config['champ_challenger']['persist_method']['module']

            if 'params' in self.config['champ_challenger']['persist_method'].keys():
                self.persist_params_champ = self.config['champ_challenger']['persist_method']['params']
            else:
                self.persist_params_champ = {}

            self.persist_params_champ['tb_name'] = self.config['champ_challenger']['tb_name']
            self.tolerance_champ = self.config['champ_challenger']['tolerance_champ']
        
        self.prod = None
        
        if 'prod' in self.config.keys():
            self.logger.log('Configurando S3 client')
            self.prod = self.config['prod']
            self.prod_remote_path = self.prod['params']['remote_path']
            self.prod_mod = self.config_mod(self.prod)
        
        self.bins = None
        if 'bins' in self.config.keys():
            self.bins = self.config['bins']
        
    """
    Função para configurar um módulo
    @param dict_mod Dicionário com os parâmetros do módulo
    """
    def config_mod(self,dict_mod):
        
        prod_pack = importlib.import_module(dict_mod['package'])
        return getattr(prod_pack,dict_mod['module'])
            
            
        