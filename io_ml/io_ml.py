from abc import ABC, abstractmethod  

"""
Classe abstrata que define o escopo de entradas e saídas utilizado pelo template
"""
class IO_ML(ABC):
        
    """
    Leitura
    """
    @abstractmethod
    def read(self):
        pass
    
    """
    Escrita
    """
    @abstractmethod
    def write(self,data):
        pass
    