from melitk.connectors import BigQuery
import json
import base64
import os
import pandas as pd
from io_ml.io_ml import IO_ML
from utils.logger import Logger
from datetime import datetime

class IO_BQ(IO_ML):
    
    def __init__(self,tb_name):
        self.logger = Logger(self)
        self.tb_name = tb_name
        credentials = json.loads(base64.b64decode(os.environ["SECRET_DS_GPC_KEY"]))
        self.bigquery = BigQuery(
                credentials=credentials
            )
    
    def read(self,query):
        
        self.logger.log('QUERY READ: {query}'.format(query=query))
        
        df = self.bigquery.execute_response(query,output='df')
        
        self.logger.log('DF: {df}'.format(df=df))
        self.logger.log('DF COLS: {cols}'.format(cols=df.columns))
        
        return df
        
    def write(self,data):      
        
        data.to_csv('data_tmp.csv',index=False, header=False)
        self.bigquery.fast_load('data_tmp.csv','{tb_name}'.format(tb_name=self.tb_name))
        os.system('rm -f data_tmp.csv')