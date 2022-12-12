from datetime import datetime 

"""
Classe geradora de logs
"""
class Logger():
    
    """
    Construtor
    @param module Módulo que instanciou a classe
    """
    def __init__(self,module):
        self.module = module.__class__
    
    """
    Imprime o log
    @param msg Mensagem a ser imprimida
    """
    def log(self,msg: str):
        
        time = datetime.now().strftime('%b %d %Y %H:%M:%S')
        print('{time} -- {mod} {msg}'.format(time=time, 
                                             mod=self.module, 
                                             msg=msg)
             )