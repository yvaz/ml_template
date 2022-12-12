from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rdelta

"""
Classe que compõe métodos estáticos úteis para tratamento de data
"""
class DateUtils():
    
    """
    Adiciona período (semana/mês) à uma data
    @param safra Safra sobre a qual será adicionado um período
    @param deta Período a ser adicionado (int)
    @param recorr Define se período é mês (month) ou semana (week)
    @return str String com a safra tratada
    """
    @staticmethod
    def add(safra,delta,recorr='month'):
        
        safra_result = None
        
        if recorr == 'month':
            safra_t = dt.strptime(safra,'%Y%m')
            safra_add = safra_t + rdelta(months=delta)
            safra_result = safra_add.strftime('%Y%m')
        
        elif recorr == 'week':
            safra_t = dt.strptime(safra,'%Y-%m-%d')
            safra_add = safra_t + rdelta(weeks=delta)
            safra_result = safra_add.strftime('%Y-%m-%d')
        
        else:
            print('RECORRÊNCIA NÃO IMPLEMENTADA')
            
        return safra_result
            