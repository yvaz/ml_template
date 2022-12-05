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
from io_ml import io_metadata
from engine.main_cfg import MainCFG

class Evaluer():
    
    def __init__(self,
                 safra: int,
                 config: str = os.path.dirname(__file__)+"/main_cfg.yaml"):

        self.logger = Logger(self)
        self.metadata = io_metadata.IOMetadata()
        self.main_cfg = MainCFG(config)
        self.config = self.main_cfg.config
        self.safra = safra
        
    def _load_score_class(self):
        
        mod = importlib.import_module(self.main_cfg.persist_package_score)
        io = getattr(mod,self.main_cfg.persist_module_score)    
            
        io_c = io(**self.main_cfg.persist_params_score)
        
        if self.main_cfg.persist_package_score == 'io_ml.io_bq':
            
            safra = du.DateUtils.add(self.safra,1)
            safra_fmt = datetime.strptime(safra,'%Y%m').strftime('%Y-%m-%d')
            query = """
                        SELECT CUS_CUST_ID,SCORES_0,SCORES_1,DECIL,DT_EXEC,DT_TRAIN
                        FROM {tb_name}
                        WHERE MODEL_NAME='{model_name}'
                        AND SAFRA='{safra}'
                        AND DT_EXEC=(
                                    SELECT max(DT_EXEC) FROM {tb_name}
                                    WHERE MODEL_NAME='{model_name}'
                                    AND SAFRA='{safra}'
                                    )
                    """.format(tb_name=self.main_cfg.persist_params_score['tb_name'],
                               safra=safra_fmt,
                               model_name=self.main_cfg.model_name)
            
            self.preds = io_c.read(query)
        else:
            self.preds = io_c.read()

    def _load_score_regr(self):
        
        mod = importlib.import_module(self.main_cfg.persist_package_score)
        io = getattr(mod,self.main_cfg.persist_module_score)    
            
        io_c = io(**self.main_cfg.persist_params_score)
        
        if self.main_cfg.persist_package_score == 'io_ml.io_bq':
            
            safra = du.DateUtils.add(self.safra,1)
            safra_fmt = datetime.strptime(safra,'%Y%m').strftime('%Y-%m-%d')
            query = """
                        SELECT CUS_CUST_ID,SCORES,DECIL,DT_EXEC,DT_TRAIN
                        FROM {tb_name}
                        WHERE MODEL_NAME='{model_name}'
                        AND SAFRA='{safra}'
                        AND DT_EXEC=(
                                    SELECT max(DT_EXEC) FROM {tb_name}
                                    WHERE MODEL_NAME='{model_name}'
                                    AND SAFRA='{safra}'
                                    )
                    """.format(tb_name=self.main_cfg.persist_params_score['tb_name'],
                               safra=safra_fmt,
                               model_name=self.main_cfg.model_name)
            
            self.preds = io_c.read(query)
        else:
            self.preds = io_c.read()
            
    def load_score(self):
        
        if self.main_cfg.type == 'classification':
            self._load_score_class()
            
        if self.main_cfg.type == 'regression':
            self._load_score_regr()

    def _evaluate_class(self,y,path):
        
        if not os.path.isdir(path):
            os.system('mkdir '+path)
            
        eval_base = y.merge(self.preds,'inner','CUS_CUST_ID')
        eval_base['target'] = eval_base['target'].fillna(0)

        pc.PlotsCollection.targets_plot(eval_base['SCORES_1'],eval_base['target'])
        plt.savefig(path+'targets_plot.png')
        plt.close()

        conversion = rc.ResultsCollection.lift(eval_base['target'],eval_base[['SCORES_0','SCORES_1']],10)
        conversion.to_csv(path+'conversion_report.csv')
                                 
        io = self.main_cfg.config_mod(self.main_cfg.persist_method_eval)
        
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
        conv['DT_TRAIN'] = self.preds['DT_TRAIN']
        
        conv = pd.concat(
            [pd.DataFrame(self.main_cfg.model_name,index=range(conv.shape[0]),columns=['MODEL_NAME'])\
                 .reset_index(drop=True)
             ,conv],
            axis=1
        )
        
        self.logger.log(conv.columns)
        
        conv.columns = ['MODEL_NAME','DECIL','PUBLICO','CONVERSAO','RESP_RATE',
                              'LIFT','CONVERSAO_ACC','CONVERSAO_PERC','KS',
                              'SAFRA','DT_EXEC','DT_TRAIN']
        self.logger.log(conv)
        
        io_c = io(**self.main_cfg.persist_params_eval)
        io_c.write(conv)
 
    def _evaluate_regr(self,y,path):
        
        if not os.path.isdir(path):
            os.system('mkdir '+path)
            
        eval_base = y.merge(self.preds,'inner','CUS_CUST_ID')
        eval_base['target'] = eval_base['target'].fillna(0)

        pc.PlotsCollection.reg_sort_plot(eval_base['SCORES'],eval_base['target'])
        plt.savefig(path+'targets_plot.png')
        plt.close()

        conversion = rc.ResultsCollection.confusion_matrix(eval_base['target'],
                                                           eval_base[['SCORES']],
                                                           self.main_cfg.bins)
        pivot_conv = conversion.pivot(index='pred_grp',columns=['target_grp'],values='cnt')
        pivot_conv.to_csv(path+'conversion_report.csv')
           
        conversion['SAFRA'] = datetime.strptime(du.DateUtils.add(self.safra,1),'%Y%m')
        conversion['DT_EXEC'] = datetime.now().strftime('%Y%m%d%H%M%S')
        conversion['DT_TRAIN'] = self.preds['DT_TRAIN']
        
        conv = pd.concat(
            [pd.DataFrame(self.main_cfg.model_name,index=range(conversion.shape[0]),columns=['MODEL_NAME'])\
                 .reset_index(drop=True)
             ,conversion],
            axis=1
        )
        
        #self.logger.log(conv.columns)
        
        conv.columns = ['MODEL_NAME','PRED_BIN','TARGET_BIN','QTDE','SAFRA',
                              'DT_EXEC','DT_TRAIN']
        self.logger.log(conv)
                                  
        io = self.main_cfg.config_mod(self.main_cfg.persist_method_eval)     
        io_c = io(**self.main_cfg.persist_params_eval)
        io_c.write(conv)       
        
    def evaluate(self,y,path):
        
        if self.main_cfg.type == 'classification':
            self._evaluate_class(y,path)
        
        if self.main_cfg.type == 'regression':
            self._evaluate_regr(y,path)
                    
        
        
        

