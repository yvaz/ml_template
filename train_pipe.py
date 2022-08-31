from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import yaml
from yaml.loader import SafeLoader
import importlib
import imp
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import csv
import pandas as pd
from plots_collection import calibration_plot

class TrainPipe(BaseEstimator, TransformerMixin):

    tuning_f = {'trial.suggest_loguniform',
                'trial.suggest_categorical',
                'trial.suggest_int'}

    def __init__(self,
                 config: str = "train_cfg.yaml"):

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)

        self.test_sample_rate = self.config['test_sample_rate']
        self.cal_sample_rate = self.config['calibration']['cal_sample_rate']
        self.tuning_sample = self.config['tuning']['sample_rate']
        self.rpt = self.config['report']
        self.calibration = 'calibration' in self.config.keys()
        self.tuning = 'tuning' in self.config.keys()



    def _tuning_train(self, X, y):

        def objective(trial, data=X, target=y):
     
            tuning_aux = self.tuning_cfg.copy()

            for trial_f in list(set(self.tuning_cfg.keys()).intersection(set(self.tuning_f))):

                func = eval(trial_f)
                params = self.tuning_cfg[trial_f]

                for p in params.keys():
                    if trial_f != 'trial.suggest_categorical':
                        tuning_aux[p] = func(p,*params[p])
                    else:
                        tuning_aux[p] = func(p,params[p])

                tuning_aux.pop(trial_f)

            train_X, test_X, train_y, test_y = train_test_split(data, target, 
                                                                test_size=self.tuning_sample,
                                                                random_state=42)

            model = self.clf(**tuning_aux)
            model.fit(train_X,train_y)
            preds = model.predict(test_X)
            metric = self.tuning_metric(test_y,preds)
            
            return metric

        self.tuning_cfg = self.config['tuning']
        self.direction = self.tuning_cfg['direction']
        self.n_trials = self.tuning_cfg['n_trials']
        self.tuning_cfg.pop('direction')
        self.tuning_cfg.pop('n_trials')
        self.tuning_cfg.pop('sample_rate')

        if 'package' in self.tuning_cfg['tuning_metric'].keys():

            pack = importlib.import_module(self.tuning_cfg['tuning_metric']['package'])
            self.tuning_metric = getattr(pack,self.tuning_cfg['tuning_metric']['module'])
            self.tuning_cfg.pop('tuning_metric')

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials) 
        print('Best trial:', study.best_trial.params)

        return study.best_trial.params
            

    def fit(self, X, y = None):
      
        pack = importlib.import_module(self.config['train']['package'])
        mod = getattr(pack,self.config['train']['module'])
        self.clf = mod

        train_X, train_y = X, y

        if self.calibration:
            
            train_X, cal_X, train_y, cal_y = train_test_split(train_X, train_y, 
                                                                test_size=self.cal_sample_rate,
                                                                random_state=42)
            self.cal_method = self.config['calibration']['method']


        if self.tuning:
        
            best_trial_params = self._tuning_train(train_X, train_y)
            self.model = self.clf(**best_trial_params)
            self.model.fit(train_X,train_y)

        else:

            self.model = self.clf(**self.config['train']['params'])

        if self.calibration:
        
           print("INICIANDO CALIBRAÇÃO") 
           calibrated_clf = CalibratedClassifierCV(
                                base_estimator=self.model,
                                cv=self.config['calibration']['cv'],
                                method=self.cal_method) 

           self.base_model = self.model
           self.model = calibrated_clf.fit(cal_X,cal_y)

        return self

    def report(self,y):
        
        if self.rpt:
            preds = self.preds
            proba = self.proba[:,1]

            result = pd.DataFrame.from_dict(classification_report(y,preds,output_dict=True))
            result.to_csv('results.csv')

            if self.calibration:
                mod = self.base_model
                calibration_plot([(mod,'Descalibrado'),
                                  (self.model,'Calibrado')],
                                  self.X_test,
                                  y)
                plt.savefig('cal_curve.png')
                plt.close()
            else:
                mod = self.model

            feature_importances = pd.DataFrame.from_dict(mod.feature_importances_)
            feature_importances.to_csv('feat_importances.csv')

    def transform(self, X, y = None):

        self.X_test = X
        self.preds = self.model.predict(X)
        self.proba = self.model.predict_proba(X)

        return self.preds, self.proba


