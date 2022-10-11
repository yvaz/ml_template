from abc import ABC, abstractmethod

class IO_ML(ABC):
        
    @abstractmethod
    def read(self):
        pass
    
    @abstractmethod
    def write(self,data):
        pass
    