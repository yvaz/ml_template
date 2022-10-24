from datetime import datetime 

class Logger():
    
    def __init__(self,module):
        self.module = module.__class__
    
    def log(self,msg: str):
        
        time = datetime.now().strftime('%b %d %Y %H:%M:%S')
        print('{time} -- {mod} {msg}'.format(time=time, 
                                             mod=self.module, 
                                             msg=msg)
             )