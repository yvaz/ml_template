import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from io_ml.io_ml import IO_ML

class IOParquet(IO_ML):
    
    def __init__(self,path,tb_name):
        self.path = path+tb_name
    
    def read(self):
        return pd.read_parquet(self.path, engine='pyarrow')
    
    def write(self,data):        
        table = pa.Table.from_pandas(data)
        pq.write_table(table, self.path)