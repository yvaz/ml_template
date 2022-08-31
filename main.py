import etl_pipe as etlp
import prep_pipe as prepp
import train_pipe as trainp
from sklearn.pipeline import Pipeline
import pickle
import sys
import yaml
from yaml.loader import SafeLoader
from sklearn.model_selection import train_test_split

def train(model_name, safra, train_test_sample):

    # PREPARANDO PIPELINE
    etl = etlp.ETLPipe(safra)
    prep = prepp.PrepPipe()
    train = trainp.TrainPipe()
    pipe = Pipeline([('prep',prep),('train',train)])
    
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
    train.report(test_y)
   
def score(model_name,safra):
    print("NÃO IMPLEMENTADO")
    
    with open(model_name+'.pkl','rb') as fp:
        model = pickle.load(fp)

    return model
    

def main(argv):

    safra = 202204

    args = argv[1:]
    config = args[0]
    is_train = args[1]

    with open(config,'r') as fp:
        config = yaml.load(fp, Loader = SafeLoader)
    
    model_name = config['model_name']
    train_test_sample = config['train_test_sample']

    if is_train:
        train(model_name,safra,train_test_sample)

    else:
        score(model_name,safra,train_test_sample)

if __name__ == '__main__':
    main(sys.argv)
