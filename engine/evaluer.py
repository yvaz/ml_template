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
from utils.logger import Logger
from utils import date_utils as du
from matplotlib import pyplot as plt
from datetime import datetime
import decimal

class Evaluer():

    def __init__(self,
                 safra: int,
                 config: str = os.path.dirname(__file__)+"/main_cfg.yaml"):

        self.logger = Logger(self)
        
        with open(config,'r') as fp:
            self.config = yaml.load(fp, Loader = SafeLoader)
                 
        self.safra = safra
        self.model_name = self.config['model_name']  
        
        self.persist_package_score = self.config['score']['persist_method']['package']
        self.persist_module_score = self.config['score']['persist_method']['module']
        
        if 'params' in self.config['score']['persist_method'].keys():
            self.persist_params_score = self.config['score']['persist_method']['params']
        else:
            self.persist_params_score = {}
            
        self.persist_params_score['tb_name'] = self.config['score']['tb_name']
        
        self.persist_package_eval = self.config['eval']['persist_method']['package']
        self.persist_module_eval = self.config['eval']['persist_method']['module']
        
        if 'params' in self.config['eval']['persist_method'].keys():
            self.persist_params_eval = self.config['eval']['persist_method']['params']
        else:
            self.persist_params_eval = {}
            
        self.persist_params_eval['tb_name'] = self.config['eval']['tb_name']
        

    def load_score(self):
        
        mod = importlib.import_module(self.persist_package_score)
        io = getattr(mod,self.persist_module_score)    
            
        io_c = io(**self.persist_params_score)
        
        if self.persist_package_score == 'io_ml.io_bq':
            
            safra = du.DateUtils.add(self.safra,1)
            safra_fmt = datetime.strptime(safra,'%Y%m').strftime('%Y-%m-%d')
            query = """
                        SELECT CUS_CUST_ID,SCORES_0,SCORES_1,DECIL
                        FROM {tb_name}
                        WHERE MODEL_NAME='{model_name}'
                        AND SAFRA='{safra}'
                        AND DT_EXEC=(
                                    SELECT max(DT_EXEC) FROM {tb_name}
                                    WHERE MODEL_NAME='{model_name}'
                                    AND SAFRA='{safra}'
                                    )
                    """.format(tb_name=self.persist_params_score['tb_name'],
                               safra=safra_fmt,
                               model_name=self.model_name)
            
            self.preds = io_c.read(query)
        else:
            self.preds = io_c.read()

    def evaluate(self,y,path):
                    
        if not os.path.isdir(path):
            os.system('mkdir '+path)
            
        eval_base = y.merge(self.preds,'inner','CUS_CUST_ID')
        eval_base['target'] = eval_base['target'].fillna(0)

        pc.PlotsCollection.roc_curve_plot(eval_base['SCORES_1'],eval_base['target'])
        plt.savefig(path+'roc_curve.png')
        plt.close()

        pc.PlotsCollection.targets_plot(eval_base['SCORES_1'],eval_base['target'])
        plt.savefig(path+'targets_plot.png')
        plt.close()

        conversion = rc.ResultsCollection.lift(eval_base['target'],eval_base[['SCORES_0','SCORES_1']],10)
        conversion.to_csv(path+'conversion_report.csv')
                                 
        mod = importlib.import_module(self.persist_package_eval)
        io = getattr(mod,self.persist_module_eval)      
        
        context = decimal.Context(prec=7)
        
        conv = conversion[['scr_grp','Negatives','Positives','resp_rate','lift','cmltv_p_perc']]
        conv['Negatives'] = conv['Negatives'] + conv['Positives']
        conv['resp_rate'] = conv['resp_rate'].apply(context.create_decimal_from_float)
        conv['lift'] = conv['lift'].apply(context.create_decimal_from_float)
        conv['cmltv_p_perc'] = conv['cmltv_p_perc'].apply(context.create_decimal_from_float)
        conv['CONVERSAO_PERC'] = (conv['Positives']/(conv['Positives'].sum())*100).apply(context.create_decimal_from_float)
        conv['KS'] = conversion['KS'].apply(context.create_decimal_from_float)
        conv['SAFRA'] = datetime.strptime(du.DateUtils.add(self.safra,1),'%Y%m')
        conv['DT_EXEC'] = datetime.now().strftime('%Y%m%d%H%M%S')
        
        self.logger.log(conv.columns)
        
        
        conv.columns = ['DECIL','PUBLICO','CONVERSAO','RESP_RATE',
                              'LIFT','CONVERSAO_ACC','CONVERSAO_PERC','KS',
                              'SAFRA','DT_EXEC']
        
        io_c = io(**self.persist_params_eval)
        self.preds = io_c.write(conv)
        
        

