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

class ETL():

    def __init__(self,
                 safra: int,
                 config: str = "core/etl_cfg.yaml", 
                 flow: str = "train",
                 labeled: bool = True,
                 persist=True):

        self.safra = safra

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
    

        self.key = self.config['key']
        self.flow = flow
        self.labeled = labeled

        self.n_samples = self.config['train_flow']['n_samples']
            
        self.target_advance = self.config['target_advance']
        self.persist = persist
        self.imbalanced = 'class_balance' in self.config['train_flow'].keys()

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

    def setup(self):

        if self.imbalanced:
            mod_class_blce = importlib.import_module(self.class_blce_mod)
            self.class_blce = getattr(mod_class_blce,self.class_blce_met)

        pub_mod = importlib.import_module(self.pub_mod)
        self.pubs = getattr(pub_mod,self.pub_met)

        if self.labeled:

            target_mod = importlib.import_module(self.target_mod)
            self.targets = getattr(target_mod,self.target_met)

        feats_mod = importlib.import_module(self.feats_mod)
        self.feats = getattr(feats_mod,self.feats_met)

    def _extract_train(self):

        if self.labeled:
            features = self.feats(self.safra)
            target = self.targets(self.safra+self.target_advance)
            pub = self.pubs(self.safra)

            if self.drop:
                features = features.drop(self.drop, axis=1)
                
            master = pub.merge(features,how='left',on=self.key)\
                            .merge(target,how='left',on=self.key)

            master['target'] = master['target'].fillna(0)

            if self.n_samples != 'None':
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
            
            if self.persist:
                io_parquet = IOParquet('registries/','train_dataset.parquet')
                io_parquet.write(master)
            
            return df_X, df_y

        else:
            features = self.features(self.safra)
            pub = self.pub(self.safra)

            if self.drop:
                features.drop(self.drop, axis=1)

            master = pub.merge(features,how='left',on='user_id')

            master = master.sample(n_sample)
            
            if self.persist:
                self._persist(master,'train_dataset.parquet')

            return master

    def _extract_score(self):
        
        features = self.feats(self.safra+1)
        pub = self.pubs(self.safra+1)

        if self.drop:
            features = features.drop(self.drop, axis=1)

        master = pub.merge(features,how='left',on=self.key)

        if self.persist:
            self._persist(master,'score_dataset.parquet')

        return master

    def _extract_eval(self):
        
        targets = self.targets(self.safra+2)
        pub = self.pubs(self.safra+1)

        master = pub.merge(targets,how='left',on=self.key)
        master['target'] = master['target'].fillna(0)
        
        if self.persist:
            self._persist(master,'eval_dataset.parquet')

        return master

    def _persist(self,dset,fname):
        
        if not exists('registries'):
            
            os.system('mkdir registries')
            
        io_parquet = IOParquet('registries/',fname)
        io_parquet.write(dset)
            
    
    def extract(self):

        if self.flow == 'train':

            return self._extract_train()

        elif self.flow == 'score':

            return self._extract_score()
        
        elif self.flow == 'eval':
            
            return self._extract_eval()

