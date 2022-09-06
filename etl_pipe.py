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
    

        self.key = self.config['key']
        self.train_flow = train_flow
        self.labeled = labeled

        self.n_samples = self.config['train_flow']['n_samples']
        self.target_advance = self.config['target_advance']
        self.imbalanced = (self.config['train_flow']['class_balance'] != None)

        if self.imbalanced:
            self.class_blce_mod = list(self.config['train_flow']['class_balance'].keys())[0]
            self.class_blce_met = self.config['train_flow']['class_balance'][self.class_blce_mod]

        self.score_method = self.config['score_flow']['method']
        
        if 'drop'in self.config.keys():
            self.drop = self.config['drop']
        else:
            self.drop = None

        self.pub_mod = list(self.config['pub'].keys())[0]
        self.pub_met = self.config['pub'][self.pub_mod]

        if self.labeled:
            self.target_mod = list(self.config['target'].keys())[0]
            self.target_met = self.config['target'][self.target_mod]

        self.feats_mod = list(self.config['features'].keys())[0]
        self.feats_met = self.config['features'][self.feats_mod]

    def fit(self):

        if self.imbalanced:
            mod_class_blce = importlib.import_module(self.class_blce_mod)
            self.class_blce = getattr(mod_class_blce,self.class_blce_met)

        mod_name,file_ext = self.pub_mod.split('.')

        pub_mod = imp.load_source(mod_name, self.pub_mod)
        self.pubs = getattr(pub_mod,self.pub_met)

        if self.labeled:
            mod_name,file_ext = self.target_mod.split('.')

            target_mod = imp.load_source(mod_name, self.target_mod)
            self.targets = getattr(target_mod,self.target_met)

        mod_name,file_ext = self.feats_mod.split('.')

        feats_mod = imp.load_source(mod_name, self.feats_mod)
        self.feats = getattr(feats_mod,self.feats_met)

    def _transform_train(self):

        if self.labeled:
            features = self.feats(self.safra)
            target = self.targets(self.safra+1)
            pub = self.pubs(self.safra)

            if self.drop:
                features = features.drop(self.drop, axis=1)

            master = pub.merge(features,how='left',on=self.key)\
                            .merge(target,how='left',on=self.key)

            master['target'] = master['target'].fillna(0)

            master = master.sample(self.n_samples)
            master = master.drop(self.key, axis=1)

            X = master.loc[:,master.columns != 'target']
            y = master['target']

            if self.imbalanced:
                blce = self.class_blce()
                df_X, df_y = blce.fit_resample(X, y)
            else:
                df_X = X
                df_y = y
            return df_X, df_y

        else:
            features = self.features(self.safra)
            pub = self.pub(self.safra)

            if self.drop:
                features.drop(self.drop, axis=1)

            master = pub.merge(features,how='left',on='user_id')

            master = master.sample(n_sample)

            return master

    #TODO
    def _transform_score(self):
        pass

    def transform(self):

        if self.train_flow:

            return self._transform_train()

        else:

           return self._transform_score()

