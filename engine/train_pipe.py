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
from sklearn.metrics import roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error
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
from utils.logger import Logger
from engine.main_cfg import MainCFG

class TrainPipe(BaseEstimator, TransformerMixin):

    tuning_f = {'trial.suggest_loguniform',
                'trial.suggest_categorical',
                'trial.suggest_int',
                'trial.suggest_uniform'}
    
    metadata_key = 'train_pipe'

    def __init__(self,
                 config: str = os.path.dirname(__file__)+"/train_cfg.yaml"):

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
            
        self.main_cfg = MainCFG()
        self.logger = Logger(self)
        self.logger.log('Inicializando pipeline de treinamento')
        
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
            
            if self.main_cfg.type == 'predict_proba':
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
        
        self.logger.log('Initializando treinamento')
        
        pack = importlib.import_module(self.config['train']['package'])
        mod = getattr(pack,self.config['train']['module'])
        self.clf = mod

        train_X = X
        train_y = y
        
        if self.calibration or self.optimize_thresh:
            
            self.logger.log('Split de dados para calibração/otimização de threshold')
            train_X, cal_X, train_y, cal_y = train_test_split(train_X, train_y, 
                                                                test_size=self.opt_sample_rate,
                                                                random_state=42)
            
            if self.calibration:
                self.cal_method = self.config['calibration']['method'] 
        
        if self.tuning:
        
            self.logger.log('TUNING')
            best_trial_params = self._tuning_train(train_X, train_y)
            params = self.train_params | best_trial_params 
            self.model = self.clf(**params)
            self.model.fit(train_X,train_y)

        else:

            self.model = self.clf(**self.config['train']['params'])

        if self.calibration:
            
            self.logger.log('CALIBRATION')
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
            
            self.logger.log('Otimização de threshold')
            thresh_clf = OptThreshold(self.model)
            self.model = thresh_clf.fit(cal_X,cal_y)
            
        meta = [
                  {'feat_dim':list(train_X.shape)},
                  {'class_proportion':len(train_y[train_y == 1])/len(train_y)}
                ]
        
        self.metadata.meta_by_list(self.metadata_key,meta)
        self.metadata.write()

        return self

    def _report_class(self,y,path):
        
        self.logger.log('Inicializando o processo de report')

        if not os.path.isdir(path):
            os.system('mkdir '+path)

        preds = self.preds
        proba = self.proba

        self.logger.log('Produzindo relatório geral')
        result = pd.DataFrame.from_dict(classification_report(y,preds,output_dict=True))
        result.to_csv(path+'results.csv')

        if self.calibration:
            self.logger.log('Produzindo calibration plot')
            mod = self.base_model
            pc.PlotsCollection.calibration_plot(proba[:,1],y)
            plt.savefig(path+'cal_curve.png')
            plt.close()
        else:
            self.logger.log('Produzindo gráfico de shap')
            mod = self.model
            tree_explainer = shap.Explainer(self.model)
            shap_values = tree_explainer.shap_values(self.X_test)
            shap.summary_plot(shap_values, self.X_test, 
                              feature_names=self.metadata.metadata['prep_pipe']['var_names'])       
            plt.savefig(path+'shap_importance.png')
            plt.close()

        self.logger.log('Produzindo gráfico de curva ROC')
        pc.PlotsCollection.roc_curve_plot(proba[:,1],y)
        plt.savefig(path+'roc_curve.png')
        plt.close()

        self.logger.log('Produzindo resultado ROC AUC')
        fpr, tpr, _ = roc_curve(y, proba[:,1])
        ras = auc(fpr, tpr)
        result = pd.DataFrame([ras])
        result.to_csv(path+'metric.csv',index=False)
        self.logger.log('-- {ras}'.format(ras=ras))

        self.logger.log('Produzindo gráfico de targets')
        pc.PlotsCollection.targets_plot(proba[:,1],y)
        plt.savefig(path+'targets_plot.png')
        plt.close()

        conversion = rc.ResultsCollection.lift(y,proba,10)
        conversion.to_csv(path+'conversion_report.csv')  

    def _report_regr(self,y,path):
        
        self.logger.log('Inicializando o processo de report')

        if not os.path.isdir(path):
            os.system('mkdir '+path)

        preds = self.preds
        
        self.logger.log('Produzindo r2')
        r2 = r2_score(preds,y)
        
        self.logger.log('Produzindo MAE')
        mae = mean_absolute_error(preds, y)
        
        self.logger.log('Produzindo MSE')
        mse = mean_squared_error(preds, y)
        
        
        self.logger.log(preds)
        self.logger.log(y)
        
        corr_df = pd.DataFrame({'PRED':preds,'TRUE':y})

        self.logger.log(corr_df.astype(float))
        self.logger.log('Produzindo pearson')
        pear = corr_df.corr(method='pearson').iloc[0,0]
        
        self.logger.log('Produzindo spearman')
        spear = corr_df.corr(method='spearman').iloc[0,0]
        
        self.logger.log(pear)
        self.logger.log(spear)
        
        results = pd.DataFrame({'R2':[r2],'MAE':[mae],'MSE':[mse],
                                'PEARSON':[pear],'SPEARMAN':[spear]})
        
        results.to_csv(path+'results.csv')
        
        pc.PlotsCollection.reg_sort_plot(preds,y)
        plt.savefig(path+'target_plot.png')
        plt.close()
        
        
    def report(self,y,path):
        
        if self.rpt:
            
            if self.main_cfg.type == 'classification':
                
                self._report_class(y,path)
            
            elif self.main_cfg.type == 'regression':
                
                self._report_regr(y,path)

    def transform(self, X, y = None):
    
        self.X_test = X
        self.preds = self.model.predict(X)
        
        if self.main_cfg.type == 'classification':
            self.proba = self.model.predict_proba(X)
            return self.preds,self.proba
        elif self.main_cfg.type == 'regression':
            return self.preds


