import engine.etl as etlp
import engine.prep_pipe as prepp
import engine.train_pipe as trainp
import engine.scorer as scorep
import engine.evaluer as evalp
from sklearn.pipeline import Pipeline
import pickle
import sys
import yaml
from yaml.loader import SafeLoader
from sklearn.model_selection import train_test_split
from os.path import exists
import pandas as pd
from io_ml.io_parquet import IOParquet
from datetime import datetime
import os
import importlib
from utils.logger import Logger
import shutil
from io_ml import io_metadata
from engine.main_cfg import MainCFG

class Executor():
    
    metadata_key = 'executor'
    
    def __init__(self,flow_type: str
                     ,safra: int
                     ,config: str = os.path.dirname(__file__)+"/main_cfg.yaml"):
        
        
        self.main_cfg = MainCFG(config)
        self.logger = Logger(self)
        self.metadata = io_metadata.IOMetadata()
        self.flow_type = flow_type
        self.safra = safra
        
    def write_pickle(self
                     ,pipe: Pipeline):
        
        path = 'registries/pkl_{safra}'.format(safra=self.safra)
        current_date = self.metadata.metadata['executor']['train_timestamp']
        
        if exists(path):
                    
            with open('{path}/{model_name}_{current_date}.pkl'.format(path=path,
                                                                      model_name=self.main_cfg.model_name,
                                                                      current_date=current_date),'wb') as fp:
                pickle.dump(pipe,fp)
        
        else:

            os.system('mkdir {path}'.format(path=path))

            with open('{path}/{model_name}_{current_date}.pkl'.format(path=path,
                                                                  model_name=self.main_cfg.model_name,
                                                                  current_date=current_date),'wb') as fp:
                pickle.dump(pipe,fp)
        
        if self.main_cfg.prod:      
            
            self.main_cfg.prod['params']['remote_path'] = self.main_cfg.prod_remote_path + \
                                                          self.main_cfg.model_name+'/{path}/{model_name}_{current_date}.pkl'\
                                                                    .format(path=path,
                                                                            model_name=self.main_cfg.model_name,
                                                                            current_date=current_date)
            
            self.main_cfg.prod['params']['local_path'] = '{path}/{model_name}_{current_date}.pkl'\
                                                                    .format(path=path,
                                                                            model_name=self.main_cfg.model_name,
                                                                            current_date=current_date)
            
            self.logger.log('Carregando pickle de S3')
            self.prod_obj = self.main_cfg.prod_mod(**self.main_cfg.prod['params'])
            
            self.prod_obj.write()
    
    def execute(self):
        
        if self.flow_type == 'train':
            self._exec_flow(self._train,'train')
        elif self.flow_type == 'score':
            self._exec_flow(self._score,'score')
        elif self.flow_type == 'eval':
            self._exec_flow(self._evaluation,'eval')
        else:
            print('TIPO NÃO PERMITIDO')
    
    def _exec_flow(self,func,flow):
        
        if flow == 'train':
            fname = 'train_dataset.parquet'
        elif flow == 'score':
            fname = 'score_dataset.parquet'
        elif flow == 'eval':
            fname = 'eval_dataset.parquet'

        # PREPARANDO PIPELINE
        if self.main_cfg.persist == 'while_execution' or self.main_cfg.persist == 'always':
            etl = etlp.ETL(self.safra,persist=True,flow=flow)
        else:
            etl = etlp.ETL(self.safra,persist=False,flow=flow)
            
        func(fname,etl)
               
        if self.main_cfg.persist == 'while_execution':
            os.system('rm -f registries/'+fname)
        
        
    def _train(self,fname,etl):
            
        current_date = datetime.now().strftime('%Y%m%d%H%M%S')
        meta = [
                  {'train_timestamp':current_date}
                ]
        self.metadata.meta_by_list(self.metadata_key,meta)
        self.metadata.write()
            
        prep = prepp.PrepPipe()
        train = trainp.TrainPipe()
        pipe = Pipeline([('prep',prep),('train',train)])  

        if exists('registries/'+fname):
            iop = IOParquet('registries/',fname)
            dset = iop.read()
            y = dset['target']
            X = dset.drop('target',axis=1)
            train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                test_size=self.main_cfg.train_test_sample,
                                                random_state=42)
        else:       
            #EXECUTANDO ETL DE TREINAMENTO
            etl.setup()
            X,y = etl.extract()
            train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                test_size=self.main_cfg.train_test_sample,
                                                random_state=42)

        pipe.fit(train_X,train_y)

        if not exists('registries'):
            os.system('mkdir registries')
        
        self.write_pickle(pipe)

        pipe.transform(test_X)
        train.report(test_y,'train_results/')
        
        if self.main_cfg.prod:      
        
            path = 'train_results'
            shutil.make_archive(path, 'zip', path)
            
            self.main_cfg.prod['params']['remote_path'] = self.main_cfg.prod_remote_path + self.main_cfg.model_name+\
                                                '/{path}/{safra}/{model_name}_{current_date}.zip'\
                                                    .format(path=path,
                                                              safra=self.safra,              
                                                              model_name=self.main_cfg.model_name,
                                                              current_date=current_date)
            self.main_cfg.prod['params']['local_path'] = '{path}.zip'.format(path=path,
                                                                      model_name=self.main_cfg.model_name,
                                                                      current_date=current_date)
            
            self.logger.log('Carregando pickle de S3')
            self.prod_obj = self.main_cfg.prod_mod(**self.main_cfg.prod['params'])
            
            self.prod_obj.write()
            

    def _score(self,fname,etl):
            
        meta = [
                  {'score_timestamp':datetime.now().strftime('%Y%m%d%H%M%S')}
                ]
        self.metadata.meta_by_list(self.metadata_key,meta)
        self.metadata.write()

        # PREPARANDO PIPELINE
        score = scorep.Scorer(safra=self.safra)

        if exists('registries/score_dataset.parquet'):
            iop = IOParquet('registries/','score_dataset.parquet')
            X = iop.read()
        else:       
            #EXECUTANDO ETL DE ESCORAGEM
            etl.setup()
            X = etl.extract()

        score.load_model()
        score.score(X)

    def _evaluation(self,fname,etl):

        # PREPARANDO PIPELINE
        evaluation = evalp.Evaluer(safra=self.safra)

        if exists('registries/eval_dataset.parquet'):
            iop = IOParquet('registries/','eval_dataset.parquet')
            y = iop.read()
        else:       
            #EXECUTANDO ETL DE ESCORAGEM
            etl.setup()
            y = etl.extract()

        evaluation.load_score()
        evaluation.evaluate(y,'evaluation/')
    

def main(argv):

    args = argv[1:]
    config = args[0]
    flow_type = args[1]
    safra = args[2]

    exc = Executor(flow_type,safra,config)
    exc.execute()

if __name__ == '__main__':
    main(sys.argv)
