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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import csv
import pandas as pd
from collections_ml import plots_collection as pc
from collections_ml import results_collection as rc
from utils.simple_calibration import SimpleCalibration
from utils.opt_threshold import OptThreshold
import os
import shap
from io_ml import io_metadata

class TrainPipe(BaseEstimator, TransformerMixin):

    tuning_f = {'trial.suggest_loguniform',
                'trial.suggest_categorical',
                'trial.suggest_int',
                'trial.suggest_uniform'}
    
    metadata_key = 'train_pipe'

    def __init__(self,
                 config: str = "core/train_cfg.yaml"):

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
            
        self.metadata = io_metadata.IOMetadata()

        self.test_sample_rate = self.config['test_sample_rate']
        self.rpt = self.config['report']
        self.calibration = 'calibration' in self.config.keys()
        self.train_params = self.config['train']['params']
        
        if 'optimize_threshold' in self.config.keys():
            self.optimize_thresh = self.config['optimize_threshold']
        else:
            self.optimize_thresh = False
        
        if self.calibration or self.optimize_thresh:
            self.opt_sample_rate = self.config['opt_sample_rate']
        else:
            self.opt_sample_rate = None
        
        self.tuning = 'tuning' in self.config.keys()
        
        if self.tuning:
            self.tuning_sample = self.config['tuning']['sample_rate']
            self.tuning_pred = self.config['tuning']['pred_tuning']



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

            params = self.train_params | tuning_aux
            print(params)
            model = self.clf(**params)
            model.fit(train_X,train_y)
            
            f_pred = getattr(model,self.tuning_pred)
            
            if self.tuning_pred == 'predict_proba':
                preds = f_pred(test_X)[:,1]
            else:
                preds = f_pred(test_X)
                
            metric = self.tuning_metric(test_y,preds)
            
            return metric

        self.tuning_cfg = self.config['tuning']
        self.direction = self.tuning_cfg['direction']
        self.n_trials = self.tuning_cfg['n_trials']
        self.tuning_cfg.pop('direction')
        self.tuning_cfg.pop('n_trials')
        self.tuning_cfg.pop('sample_rate')
        self.tuning_cfg.pop('pred_tuning')

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

        train_X = X
        train_y = y
        
        if self.calibration or self.optimize_thresh:
            
            train_X, cal_X, train_y, cal_y = train_test_split(train_X, train_y, 
                                                                test_size=self.opt_sample_rate,
                                                                random_state=42)
            
            if self.calibration:
                self.cal_method = self.config['calibration']['method'] 
        
        if self.tuning:
        
            best_trial_params = self._tuning_train(train_X, train_y)
            params = self.train_params | best_trial_params 
            self.model = self.clf(**params)
            self.model.fit(train_X,train_y)

        else:

            self.model = self.clf(**self.config['train']['params'])

        if self.calibration:
            
            rus = SMOTE()
            cal_X,cal_y = rus.fit_resample(cal_X,cal_y)
            
            calibrated_clf = CalibratedClassifierCV(
                                base_estimator=self.model,
                                cv=self.config['calibration']['cv'],
                                method=self.cal_method)
            
            #calibrated_clf = SimpleCalibration(self.model,1000)

            self.base_model = self.model
            self.model = calibrated_clf.fit(cal_X,cal_y)
            
        if self.optimize_thresh:
            
            thresh_clf = OptThreshold(self.model)
            self.model = thresh_clf.fit(cal_X,cal_y)
            
        self.metadata.append_data(self.metadata_key,
                                  {'feat_dim':list(train_X.shape)})
        self.metadata.append_data(self.metadata_key,
                                  {'class_proportion':len(train_y[train_y == 1])/len(train_y)})
        
        print('METADATA')
        print(self.metadata)
        print(self.metadata.metadata)
        
        self.metadata.write()

        return self

    def report(self,y,path):
        
        if self.rpt:
            
            if not os.path.isdir(path):
                os.system('mkdir '+path)
            
            preds = self.preds
            proba = self.proba

            result = pd.DataFrame.from_dict(classification_report(y,preds,output_dict=True))
            result.to_csv(path+'results.csv')

            if self.calibration:
                mod = self.base_model
                pc.PlotsCollection.calibration_plot(proba[:,1],y)
                plt.savefig(path+'cal_curve.png')
                plt.close()
            else:
                mod = self.model
                tree_explainer = shap.Explainer(self.model)
                shap_values = tree_explainer.shap_values(self.X_test)
                shap.summary_plot(shap_values, self.X_test, 
                                  feature_names=self.metadata.metadata['prep_pipe']['var_names'])           
                plt.savefig(path+'shap_importance.png')
                plt.close()
            
            pc.PlotsCollection.roc_curve_plot(proba[:,1],y)
            plt.savefig(path+'roc_curve.png')
            plt.close()
            
            pc.PlotsCollection.targets_plot(proba[:,1],y)
            plt.savefig(path+'targets_plot.png')
            plt.close()
            
            conversion = rc.ResultsCollection.lift(y,proba,10)
            conversion.to_csv(path+'conversion_report.csv')

    def transform(self, X, y = None):
    
        self.X_test = X
        self.preds = self.model.predict(X)
        self.proba = self.model.predict_proba(X)

        return self.preds,self.proba


