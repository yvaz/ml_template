import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from io_ml.io_ml import IO_ML

"""
Classe que trata entrada e saída de parquets
"""
class IOParquet(IO_ML):
    
    """
    Construtor
    @param path Caminho do arquivo
    @param tb_name nome do arquivo
    """
    def __init__(self,path,tb_name):
        self.path = path+tb_name
    
    """
    Leitura do parquet
    @return pandas.DataFrame Retorna dados do parquet
    """
    def read(self):
        return pd.read_parquet(self.path, engine='pyarrow')
    
    """
    Escreve dados em parquet
    @param data pandas.DataFrame a ser escrito
    """   
    def write(self,data):        
        table = pa.Table.from_pandas(data)
        pq.write_table(table, self.path)