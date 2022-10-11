from os.path import exists
import os
import yaml
from yaml.loader import SafeLoader
from io_ml.io_ml_singleton import IO_ML_Singleton
    
class IOMetadata(IO_ML_Singleton):
    
    file_name = '.ml_metadata.yaml'
    
    def __init__(self):
        self.metadata = None
        self.read()
    
    def append_data(self,
                    key: str,
                    dict_data: dict):
        
        if not self.metadata:
            self.metadata = {key: dict_data}
        else:
            if key not in self.metadata:
                self.metadata[key] = {}
                
            for k in dict_data.keys():
                self.metadata[key][k] = dict_data[k]
                
        print('APPEND')
        print(self.metadata)
    
    def read(self):
        
        if exists(self.file_name):          
            with open(self.file_name,'r') as fp:
                self.metadata = yaml.load(fp, Loader = SafeLoader)
                fp.close()
            
        print('READ')
        print(self.metadata)
        
    def write(self):
        print('WRITE')
        print(self.metadata)
        with open(self.file_name,'w') as fp:
            yaml.dump(self.metadata, fp)
            fp.close()
        
                
                
            