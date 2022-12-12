from melitk.connectors import BigQuery
import json
import base64
import os
import pandas as pd
from io_ml.io_ml import IO_ML
from utils.logger import Logger
from datetime import datetime

"""
Classe que trata a entrada e saída para o BigQuery
"""
class IO_BQ(IO_ML):
    
    """
    Construtor
    @param tb_name Nome da tabela para busca/inserção
    """
    def __init__(self,tb_name):
        self.logger = Logger(self)
        self.tb_name = tb_name
        credentials = json.loads(base64.b64decode(os.environ["SECRET_DS_GPC_KEY"]))
        self.bigquery = BigQuery(
                credentials=credentials
            )
    
    """
    Leitura de tabela
    @param query Executa a query <query>
    @return pandas.DataFrame Dataframe com os dados da retornados pela execução de <query>
    """
    def read(self,query):
        
        self.logger.log('QUERY READ: \n{query}'.format(query=query))
        
        df = self.bigquery.execute_response(query,output='df')
        
        self.logger.log('DF:\n{df}'.format(df=df))
        self.logger.log('DF COLS: {cols}'.format(cols=df.columns))
        
        return df
        
    """
    Insere <data> na tabela self.tb_name
    @param data Dados a serem inseridos em self.tb_name
    """
    def write(self,data):      
        
        data.to_csv('data_tmp.csv',index=False, header=False)
        self.logger.log('Adding data to table {tb_name}'.format(tb_name=self.tb_name))
        self.bigquery.fast_load('data_tmp.csv','{tb_name}'.format(tb_name=self.tb_name))
        os.system('rm -f data_tmp.csv')