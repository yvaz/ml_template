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

class PrepPipe(BaseEstimator, TransformerMixin):

    estimator_params = {'boruta':['estimator']}
    eval_params = ['missing_values']
    
    metadata_key = 'prep_pipe'

    def __init__(self,
                 config: str = os.path.dirname(__file__)+"/prep_cfg.yaml"):

        self.config_name = config
        
        with open(self.config_name,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)

        self.logger = Logger(self)
        self.metadata = io_metadata.IOMetadata()
        self.cat_steps = self.config['categorical']['steps']
        self.num_steps = self.config['numerical']['steps']
        self.final_steps = self.config['final']['steps']

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


    def fit(self, X, y = None):
       
        cats = self.config['categories']
        nums = list(set(X.columns) - set(cats))

        
        self.logger.log('Definindo configurações à partir do arquivo {cfg}'.format(cfg=self.config_name))
        

        self.logger.log('Configurando pré-processamento de variáveis categóricas')
        cat_pipe = Pipeline(self._create_pipe(self.cat_steps))

        self.logger.log('Configurando pré-processamento de variáveis numéricas')
        num_pipe = Pipeline(self._create_pipe(self.num_steps))

        self.logger.log('Configurando fluxo final')
        final_pipe = Pipeline(self._create_pipe(self.final_steps))

        col_transformer = ColumnTransformer(
                    transformers=[
                        ('cat',cat_pipe,cats),
                        ('num',num_pipe,nums)
                        ]
        )

        X[nums] = X[nums].astype(float)

        self.logger.log('Construindo pipeline')
        self.pipe = Pipeline([('preproc',col_transformer),('final',final_pipe)])
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

    def transform(self, X, y = None):

        return self.pipe.transform(X)
