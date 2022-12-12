from os.path import exists
import os
import yaml
from yaml.loader import SafeLoader
from io_ml.io_ml_singleton import IO_ML_Singleton
from utils.logger import Logger
    
"""
Classe que trata a entrada e saída para um arquivo de metadados de nome .ml_metadata.yaml
Essa classe é um singleton e, portanto, mudanças em quaisquer instâncias da classe serão consolidadas
Essa classe é utilizada, por exemplo, para armazenar informações úteis do pipeline de um modelo, tal como
    nome de colunas
"""
class IOMetadata(IO_ML_Singleton):
    
    # Nome do arquivo de metadado
    file_name = '.ml_metadata.yaml'
    
    """
    Construtor
    O construtor sempre lê o arquivo de metadado se houver
    """
    def __init__(self):
        self.metadata = None
        self.read()
    
    """
    Inserção de dados
    Esse método insere novos dados no arquivo de metadados
    @param key Chave em que os novos dados serão inseridos
    @param dict_data Dados (dicionários) a serem inseridos
    """
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
    
    """
    Inserção de dados
    Esse método insere uma list de dados no arquivo de metadados
    @param key Chave em que os novos dados serão inseridos
    @param meta_l Lista de dicionários a serem inseridos
    """   
    def meta_by_list(self,
                     key: str,
                     meta_l: list):
        
        for m in meta_l:
            self.append_data(key,m)
    
    """
    Lê metadados
    """
    def read(self):
        
        if exists(self.file_name):          
            with open(self.file_name,'r') as fp:
                self.metadata = yaml.load(fp, Loader = SafeLoader)
                fp.close()
    
    """
    Escreve nos metadados
    """
    def write(self):
        with open(self.file_name,'w') as fp:
            yaml.dump(self.metadata, fp)
            fp.close()
        
                
                
            