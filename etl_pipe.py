from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import yaml
from yaml.loader import SafeLoader
import importlib
import imp

class ETLPipe(BaseEstimator, TransformerMixin):

    def __init__(self,
                 safra: int,
                 config: str = "etl_cfg.yaml", 
                 train_flow: bool = True,
                 labeled: bool = True):

        self.safra = safra

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
        
        self.train_flow = train_flow
        self.labeled = labeled

        self.n_samples = self.config['train_flow']['n_samples']

        self.imbalanced = (self.config['train_flow']['class_balance'] != None)

        if self.imbalanced:
            self.class_blce_mod = list(self.config['train_flow']['class_balance'].keys())[0]
            self.class_blce_met = self.config['train_flow']['class_balance'][self.class_blce_mod]

        self.score_method = self.config['score_flow']['method']
        self.drop = self.config['drop']
        self.input_mod = list(self.config['input'].keys())[0]
        self.input_met = self.config['input'][self.input_mod]

    def fit(self):

        if self.imbalanced:
            mod_class_blce = importlib.import_module(self.class_blce_mod)
            self.class_blce = getattr(mod_class_blce,self.class_blce_met)

        mod_name,file_ext = self.input_mod.split('.')

        feats_mod = imp.load_source(mod_name, self.input_mod)
        self.feats = getattr(feats_mod,self.input_met)

    def _transform_train(self):

        if self.labeled:
            X, y = self.feats(self.safra)

            if self.imbalanced:
                under_s = self.class_blce()
                df_X, df_y = under_s.fit_resample(X, y)
            else:
                df_X = X
                df_y = Y
        else:
            df_X = self.feats(self.safra)

        return df_X, df_y

    #TODO
    def _transform_score(self):
        pass

    def transform(self):

        if self.train_flow:

            return self._transform_train()

        else:

           return self._transform_score()

