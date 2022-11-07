from utils.singleton import ABCSingleton
from abc import abstractmethod

class IO_ML_Singleton(metaclass=ABCSingleton):
        
    @abstractmethod
    def read(self):
        pass
    
    @abstractmethod
    def write(self,data):
        pass
    