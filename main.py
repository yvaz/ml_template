import etl_pipe as etlp
import prep_pipe as prepp
import train_pipe as trainp
from sklearn.pipeline import Pipeline
import pickle
import sys
import yaml
from yaml.loader import SafeLoader
from sklearn.model_selection import train_test_split
from os.path import exists
import pandas as pd

def train(model_name, safra, train_test_sample, persist):

    # PREPARANDO PIPELINE
    etl = etlp.ETLPipe(safra,persist=persist)
    prep = prepp.PrepPipe()
    train = trainp.TrainPipe()
    pipe = Pipeline([('prep',prep),('train',train)])  

    if exists('dataset.parquet'):
        dset = pd.read_parquet('dataset.parquet', engine='pyarrow')
        print("LENDO PARQUET DE DIM: {dim}".format(dim=dset.shape))
        y = dset['target']
        print("QUANTIDADE DE TARGETS: {targets}".format(targets=sum(y == 1)))
        print("QUANTIDADE DE N TARGETS: {n_targets}".format(n_targets=sum(y == 0)))
        print("TAXA TARGET {t_target}".format(t_target=sum(y == 1)/len(y)))
        X = dset.drop('target',axis=1)
        train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                            test_size=train_test_sample,
                                            random_state=42)
    else:       
        #EXECUTANDO ETL DE TREINAMENTO
        etl.fit()
        X,y = etl.transform()
        train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                            test_size=train_test_sample,
                                            random_state=42)

    pipe.fit(train_X,train_y)

    with open(model_name+'.pkl','wb') as fp:
        pickle.dump(pipe,fp)

    pipe.transform(test_X)
    print('MAIN PROBA')
    print(train.proba)
    train.report(test_y)
   
def score(model_name,safra):
    print("NÃO IMPLEMENTADO")
    
    with open(model_name+'.pkl','rb') as fp:
        model = pickle.load(fp)

    return model
    

def main(argv):

    args = argv[1:]
    config = args[0]
    is_train = args[1]
    safra = int(args[2])

    with open(config,'r') as fp:
        config = yaml.load(fp, Loader = SafeLoader)
    
    model_name = config['model_name']
    train_test_sample = config['train_test_sample']
    persist = config['persist']

    if is_train:
        train(model_name,safra,train_test_sample,persist)

    else:
        score(model_name,safra,train_test_sample)

if __name__ == '__main__':
    main(sys.argv)
