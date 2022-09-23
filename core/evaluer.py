import yaml
from yaml.loader import SafeLoader
import pickle
import pandas as pd
import numpy as np
import importlib
from collections_ml import plots_collection as pc
from collections_ml import results_collection as rc
import shap
import os
from matplotlib import pyplot as plt

class Evaluer():

    def __init__(self,
                 config: str = "core/main_cfg.yaml"):

        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
                 
        self.model_name = self.config['model_name']  
        
        self.persist_package_score = self.config['score']['persist_method']['package']
        self.persist_module_score = self.config['score']['persist_method']['module']
        self.persist_params_score = self.config['score']['persist_method']['params']
        self.persist_params_score['tb_name'] = self.config['score']['tb_name']
        
        self.persist_package_eval = self.config['eval']['persist_method']['package']
        self.persist_module_eval = self.config['eval']['persist_method']['module']
        self.persist_params_eval = self.config['eval']['persist_method']['params']
        self.persist_params_eval['tb_name'] = self.config['eval']['tb_name']
        

    def load_score(self):
                                 
        mod = importlib.import_module(self.persist_package_score)
        io = getattr(mod,self.persist_module_score)    
            
        io_c = io(**self.persist_params_score)
        self.preds = io_c.read()

    def evaluate(self,y,path):
                    
        if not os.path.isdir(path):
            os.system('mkdir '+path)
            
        eval_base = y.merge(self.preds,'left','CUS_CUST_ID')
        eval_base['target'] = eval_base['target'].fillna(0)

        pc.PlotsCollection.roc_curve_plot(eval_base['scores_1'],eval_base['target'])
        plt.savefig(path+'roc_curve.png')
        plt.close()

        pc.PlotsCollection.targets_plot(eval_base['scores_1'],eval_base['target'])
        plt.savefig(path+'targets_plot.png')
        plt.close()

        conversion = rc.ResultsCollection.lift(eval_base['target'],eval_base[['scores_0','scores_1']],10)
        conversion.to_csv(path+'conversion_report.csv')
                                 
        mod = importlib.import_module(self.persist_package_eval)
        io = getattr(mod,self.persist_module_eval)      
            
        io_c = io(**self.persist_params_eval)
        self.preds = io_c.write(conversion)
        
        

