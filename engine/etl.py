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
Classe que executa o ETL das Features, Pub Alvo, Pub Score e Targets
"""
class ETL():

    # Chave do metadado
    metadata_key = 'etl_pipe'
    
    """
    Construtor
    @param safra Safra definida para executar o ETL
    @param config Arquivo de configuração
    @param flow Fluxo de processamento ('train','score','eval')
    @param model_type Tipo de modelo ('classificação','regressão')
    @param persist Tipo de persistência dos dados de insumos ('always','while_execution','never')
    """
    def __init__(self,
                 safra: int,
                 config: str = os.path.dirname(__file__)+"/etl_cfg.yaml", 
                 flow: str = "train",
                 model_type: str = 'classification',
                 persist=True):

        self.logger = Logger(self)
        
        self.logger.log('Inicializando ETL')
        
        self.safra = safra
        self.metadata = io_metadata.IOMetadata()
        self.config_name = config

        with open(self.config_name,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
    

        self.key = self.config['key']
        self.flow = flow
        self.model_type = model_type
        self.recurrence = self.config['recurrence']

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

        if self.model_type in ('classification','regression'):
            
            self.target_mod = list(self.config['target'].keys())[0]
            self.target_met = self.config['target'][self.target_mod]
        
        self.score_tb = 'scores' in self.config.keys()
        if self.score_tb:
            self.score_mod = list(self.config['scores'].keys())[0]
            self.score_met = self.config['scores'][self.score_mod]        

        self.feats_mod = list(self.config['features'].keys())[0]
        self.feats_met = self.config['features'][self.feats_mod]

    """
    Configura os módulos a serem utilizados pelo ETL
    """
    def setup(self):

        self.logger.log('Configurando ETL a partir de {cfg}'.format(cfg=self.config_name))
        
        if self.imbalanced:
            mod_class_blce = importlib.import_module(self.class_blce_mod)
            self.class_blce = getattr(mod_class_blce,self.class_blce_met)

        pub_mod = importlib.import_module(self.pub_mod)
        self.pubs = getattr(pub_mod,self.pub_met)

        if self.model_type in ('classification','regression'):

            target_mod = importlib.import_module(self.target_mod)
            self.targets = getattr(target_mod,self.target_met)
            
        if self.score_tb:

            score_mod = importlib.import_module(self.score_mod)
            self.score_master = getattr(score_mod,self.score_met)

        feats_mod = importlib.import_module(self.feats_mod)
        self.feats = getattr(feats_mod,self.feats_met)
    
    """
    Extrai dataset de treinamento podendo ou não salvar em ../registries/
    """
    def _extract_train(self):

        if self.model_type in ('classification','regression'):
            
            self.logger.log('Extraindo dataset de treinamento para aprendizado supervisionado')
            
            features = self.feats(self.safra)
            target = self.targets(du.DateUtils.add(self.safra,self.target_advance,self.recurrence))
            pub = self.pubs(self.safra)

            if self.drop:
                features = self._drop(features)
                
            master = pub.merge(features,how='left',on=self.key)\
                            .merge(target,how='left',on=self.key)

            master['target'] = master['target'].fillna(0)

            if self.n_samples != 'None':
                self.logger.log('Selecionando subsampling do dataset')
                master = master.sample(self.n_samples)
                
            master = master.drop(self.key, axis=1)

            X = master.loc[:,master.columns != 'target']
            y = master['target']

            if self.imbalanced:
                self.logger.log('Efetuando balanceamento de classes')
                blce = self.class_blce()
                df_X, df_y = blce.fit_resample(X, y)
            else:
                df_X = X
                df_y = y
            
            if self.persist:
                self._persist(master,'train_dataset.parquet')
            
            meta = [
                      {'feat_dim':list(df_X.shape)},
                      {'class_proportion':len(df_y[df_y == 1])/len(df_y)},
                      {'var_names':df_X.columns.tolist()}
                    ]
            self.metadata.meta_by_list(self.metadata_key,meta)
            
            return df_X, df_y

        else:
                        
            self.logger.log('Extraindo dataset de treinamento para aprendizado não-supervisionado')
                                
            features = self.features(self.safra)
            pub = self.pub(self.safra)

            if self.drop:
                features = self._drop(features)

            master = pub.merge(features,how='left',on='user_id')

            master = master.sample(n_sample)
            
            if self.persist:
                self._persist(master,'train_dataset.parquet')
            
            meta = [{'feat_dim':list(master.shape)},
                    {'feat_names':master.columns.tolist()}]
            self.metadata.meta_by_list(self.metadata_key,meta)

            return master

    """
    Extrai dataset de score podendo ou não salvar em ../registries/
    """
    def _extract_score(self):
        
        self.logger.log('Extraindo dataset de escoragem')
            
        if not self.score_tb:
            self.logger.log('Público de score: PUB ALVO')
            feat_safra = du.DateUtils.add(self.safra,1,self.recurrence)
            pub_safra = du.DateUtils.add(self.safra,1,self.recurrence)

            features = self.feats(feat_safra)
            pub = self.pubs(pub_safra)

            if self.drop:
                    features = self._drop(features)

            master = pub.merge(features,how='left',on=self.key)
      
            meta = [
                      {
                          'score_safra':
                              {'feat':feat_safra,
                                'pub':pub_safra
                              },
                          'score_dim':list(master.shape)
                      }
                    ]
        else:
            self.logger.log('Público de score: PUB SCORE')
            feat_safra = du.DateUtils.add(self.safra,1,self.recurrence)
            master = self.score_master(feat_safra)
    
            meta = [
                      {
                          'score_safra':
                              {'score':feat_safra},
                          'score_dim':list(master.shape)
                      }
                    ]

        if self.persist:
            self._persist(master,'score_dataset.parquet')
        self.metadata.meta_by_list(self.metadata_key,meta)
        
        return master

    """
    Extrai dataset de avaliação podendo ou não salvar em ../registries/
    """
    def _extract_eval(self):
        
        self.logger.log('Extraindo dataset de avaliação')
        
        target_safra = du.DateUtils.add(self.safra,self.target_advance+1,self.recurrence)
        pub_safra = du.DateUtils.add(self.safra,1,self.recurrence)
                              
        targets = self.targets(target_safra)
        pub = self.pubs(pub_safra)

        master = pub.merge(targets,how='left',on=self.key)
        master['target'] = master['target'].fillna(0)
        
        if self.persist:
            self._persist(master,'eval_dataset.parquet')
        
        meta = [{
                  'eval_safra':
                      {'target':target_safra,
                        'pub':pub_safra
                      }
                }]
        self.metadata.meta_by_list(self.metadata_key,meta)

        return master

    """
    Persiste dataset em registries/
    """
    def _persist(self,dset,fname):
        
        self.logger.log('Persistindo dataset de treinamento em registries')
                        
        if not exists('registries'):
            
            os.system('mkdir registries')
            
        io_parquet = IOParquet('registries/',fname)
        io_parquet.write(dset)
                     
    """
    Dropa variáveis definidas no arquivo de configuração etl_cfg.yaml
    @param features Lista com nome das features a serem dropadas
    """
    def _drop(self,features):
                
        self.logger.log('Dropando variáveis defindas no arquivo de configuração')
        features = features.drop(self.drop, axis=1)
        return features    
                   
    """
    Executa o ETL dependendo do fluxo de processamento do modelo
    @return pandas.DataFrame Dados extraídos pelo script
    """
    def extract(self):
                              
        if self.flow == 'train':

            data = self._extract_train()

        elif self.flow == 'score':
            
            data = self._extract_score()
        
        elif self.flow == 'eval':
            
            data = self._extract_eval()
        
        self.metadata.write()
        return data

