import core.etl as etlp
import core.prep_pipe as prepp
import core.train_pipe as trainp
import core.scorer as scorep
import core.evaluer as evalp
from sklearn.pipeline import Pipeline
import pickle
import sys
import yaml
from yaml.loader import SafeLoader
from sklearn.model_selection import train_test_split
from os.path import exists
import pandas as pd
from io_ml.io_parquet import IOParquet

class Executor():
    
    def __init__(self,config,flow_type,safra):
        
        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)

        self.model_name = self.config['model_name']
        self.train_test_sample = self.config['train_test_sample']
        self.persist = self.config['persist']
        self.flow_type = flow_type
        self.safra = safra
        
    def execute(self):
        
        if self.flow_type == 'train':
            self.train()
        elif self.flow_type == 'score':
            self.score()
        elif self.flow_type == 'eval':
            self.evaluation()
        else:
            print('TIPO NÃO PERMITIDO')
        
        
    def train(self):

        # PREPARANDO PIPELINE
        etl = etlp.ETL(self.safra,persist=self.persist)
        prep = prepp.PrepPipe()
        train = trainp.TrainPipe()
        pipe = Pipeline([('prep',prep),('train',train)])  

        if exists('registries/train_dataset.parquet'):
            iop = IOParquet('registries/','train_dataset.parquet')
            dset = iop.read()
            y = dset['target']
            X = dset.drop('target',axis=1)
            train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                test_size=self.train_test_sample,
                                                random_state=42)
        else:       
            #EXECUTANDO ETL DE TREINAMENTO
            etl.setup()
            X,y = etl.extract()
            train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                test_size=self.train_test_sample,
                                                random_state=42)

        pipe.fit(train_X,train_y)

        if not exists('registries'):
            os.system('mkdir registries')
        
        with open('registries/'+self.model_name+'.pkl','wb') as fp:
            pickle.dump(pipe,fp)

        pipe.transform(test_X)
        train.report(test_y,'train_results/')

    def score(self):

        # PREPARANDO PIPELINE
        etl = etlp.ETL(self.safra,persist=self.persist,flow='score')
        score = scorep.Scorer()

        if exists('registries/score_dataset.parquet'):
            iop = IOParquet('registries/','score_dataset.parquet')
            X = iop.read()
        else:       
            #EXECUTANDO ETL DE ESCORAGEM
            etl.setup()
            X = etl.extract()

        score.load_model()
        score.score(X)

    def evaluation(self):

        # PREPARANDO PIPELINE
        etl = etlp.ETL(self.safra,persist=self.persist,flow='eval')
        evaluation = evalp.Evaluer()

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

    exc = Executor(config,flow_type,safra)
    exc.execute()

if __name__ == '__main__':
    main(sys.argv)
