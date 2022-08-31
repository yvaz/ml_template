from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import yaml
from yaml.loader import SafeLoader
import importlib
import imp
import numpy as np

class PrepPipe(BaseEstimator, TransformerMixin):

    estimator_params = {'boruta':['estimator']}
    eval_params = ['missing_values']

    def __init__(self,
                 config: str = "prep_cfg.yaml"):

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)

    def _create_pipe(self, steps):
        
        mod_list = []  

        for step in steps:

            step_name = list(step.keys())[0]
            pack = importlib.import_module(step['package'])
            mod = getattr(pack,step['module'])

            if 'params' in step:
                params = step['params']
                eval_p =  set(params).intersection(self.eval_params)
            
                for p in eval_p:
                    print(params[p])
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

        self.cat_steps = self.config['categorical']['steps']
        self.num_steps = self.config['numerical']['steps']
        self.final_steps = self.config['final']['steps']

        cat_pipe = Pipeline(self._create_pipe(self.cat_steps))
        num_pipe = Pipeline(self._create_pipe(self.num_steps))
        final_pipe = Pipeline(self._create_pipe(self.final_steps))

        col_transformer = ColumnTransformer(
                    transformers=[
                        ('cat',cat_pipe,cats),
                        ('num',num_pipe,nums)
                        ]
        )

        self.pipe = Pipeline([('preproc',col_transformer),('final',final_pipe)])
        self.pipe.fit(X,y)

        return self

    def transform(self, X, y = None):

        return self.pipe.transform(X)
