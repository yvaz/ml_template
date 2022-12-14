from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import yaml
from yaml.loader import SafeLoader
import importlib
import imp
import numpy as np
from io_ml import io_metadata
from utils.logger import Logger
import os

"""
Classe do pipeline de preprocessamento dos dados
"""
class PrepPipe(BaseEstimator, TransformerMixin):

    estimator_params = {'boruta':['estimator']}
    eval_params = ['missing_values']
    
    metadata_key = 'prep_pipe'

    """
    Construtor
    @param config Arquivo de configuração do préprocessamento dos dados
    """
    def __init__(self,
                 config: str = os.path.dirname(__file__)+"/prep_cfg.yaml"):

        self.config_name = config
        
        with open(self.config_name,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)

        self.logger = Logger(self)
        self.metadata = io_metadata.IOMetadata()
        
        self.cat_steps = None
        if 'categorical' in self.config.keys():
            self.cat_steps = self.config['categorical']['steps']
            
        self.num_steps = None
        if 'numerical' in self.config.keys():
            self.num_steps = self.config['numerical']['steps']
            
        self.final_steps = self.config['final']['steps']

    """
    Configura os módulos usados nos steps do preprocessamento
    @param steps Passos definidos no arquivo de configuração
    @return Retorna uma lista de módulos python utilizados para o preprocessamento
    """
    def _create_pipe(self, steps):
        
        mod_list = []  
        
        for step in steps:

            self.logger.log('Configurando passo: {step}'.format(step=step))
            step_name = list(step.keys())[0]
            pack = importlib.import_module(step['package'])
            mod = getattr(pack,step['module'])

            if 'params' in step:
                params = step['params']
                eval_p =  set(params).intersection(self.eval_params)
            
                for p in eval_p:
                    params[p] = eval(params[p])

                if step['package'] in self.estimator_params.keys():

                    for estm in self.estimator_params[step['package']]:

                        estm_params = step['params'][estm]
                        subpack = importlib.import_module(estm_params['package'])
                        submod = getattr(subpack,estm_params['module'])
                        subparams = estm_params['params']
                        params[estm] = submod(**subparams)

                estimator = (step_name,mod(**params))
            else:
                estimator = (step_name,mod())

            mod_list.append(estimator)

        return mod_list


    """
    Aplica o fit do preprocessamento nos dados
    @param X Features
    @param y Labels
    @return self PrepPipe
    """
    def fit(self, X, y = None):
       
        if 'categories' in self.config.keys():
            cats = self.config['categories']
        else:
            cats = {}
        nums = list(set(X.columns) - set(cats))

        
        self.logger.log('Definindo configurações à partir do arquivo {cfg}'.format(cfg=self.config_name))
        

        if self.cat_steps:      
            self.logger.log('Configurando pré-processamento de variáveis categóricas')
            cat_pipe = Pipeline(self._create_pipe(self.cat_steps))

        if self.num_steps:
            self.logger.log('Configurando pré-processamento de variáveis numéricas')
            num_pipe = Pipeline(self._create_pipe(self.num_steps))

        if self.num_steps and self.cat_steps:
            col_transformer = ColumnTransformer(
                        transformers=[
                            ('cat',cat_pipe,cats),
                            ('num',num_pipe,nums)
                            ]
            )
        elif not self.num_steps and self.cat_steps:           
            col_transformer = ColumnTransformer(
                        transformers=[
                            ('cat',cat_pipe,cats)
                            ]
            )
        elif not self.cat_steps and self.num_steps:
            col_transformer = ColumnTransformer(
                        transformers=[
                            ('num',num_pipe,nums)
                            ]
            )
        else:
            col_transformer = None

        self.logger.log('Configurando fluxo final')
        final_pipe = Pipeline(self._create_pipe(self.final_steps))

        X[nums] = X[nums].astype(float)

        self.logger.log('Construindo pipeline')
        
        if col_transformer:
            self.pipe = Pipeline([('preproc',col_transformer),('final',final_pipe)])
        else:
            self.pipe = final_pipe
            
        self.logger.log('FIT')
        self.pipe.fit(X,y)
            
        meta = [
                  {'feat_dim':list(X.shape)},
                  {'class_proportion':len(y[y == 1])/len(y)},
                  {'var_names':X.columns.tolist()}
                ]
        
        self.metadata.meta_by_list(self.metadata_key,meta)
        self.metadata.write()
        
        return self

    """
    Transforma os dados
    @param X Features
    @param y Labels
    @return numpy.array Dados tratados
    """
    def transform(self, X, y = None):

        return self.pipe.transform(X)
