import numpy as np
import pandas as pd
    

"""
Classe utilizada para compor tabela de resultados
"""    
class ResultsCollection():
    
    """
    Método que apresenta a tabela de conversão com reponse rate, lift, percentual cumulativo de conversão, KS, etc.
    @param test Label de teste
    @param pred Label predita
    @param cardinality Cardinalidade do corte dos decis (tipicamente 10)
    @return pandas.DataFrame Retorna dataframe com a avaliação do modelo
    """
    @staticmethod
    def lift (test, pred, cardinality):

        res = pd.DataFrame(np.column_stack((test, pred)),
                           columns=['Target','PR_0', 'PR_1'])

        res['scr_grp'] = pd.qcut(res['PR_0'].rank(method='first'), cardinality, labels=False)+1
        crt = pd.crosstab(res.scr_grp, res.Target).reset_index()
        crt = crt.rename(columns= {'Target':'Np',0.0: 'Negatives', 1.0: 'Positives'})

        G = crt['Positives'].sum()
        B = crt['Negatives'].sum()

        # Calculos das métricas
        avg_resp_rate = G/(G+B)

        crt['resp_rate'] = round(crt['Positives']/(crt['Positives']+crt['Negatives']),2)
        crt['lift'] = round((crt['resp_rate']/avg_resp_rate),2)
        crt['rand_resp'] = 1/cardinality
        crt['cmltv_p'] = round((crt['Positives']).cumsum(),2)
        crt['cmltv_p_perc'] = round(((crt['Positives']/G).cumsum())*100,1)
        crt['cmltv_n'] = round((crt['Negatives']).cumsum(),2)
        crt['cmltv_n_perc'] = round(((crt['Negatives']/B).cumsum())*100,1)
        crt['cmltv_rand_p_perc'] = (crt.rand_resp.cumsum())*100
        crt['cmltv_resp_rate'] = round(crt['cmltv_p']/(crt['cmltv_p']+crt['cmltv_n']),2)
        crt['cmltv_lift'] = round(crt['cmltv_resp_rate']/avg_resp_rate,2)
        crt['KS']=round(crt['cmltv_p_perc']-crt['cmltv_rand_p_perc'],2)

        crt = crt.drop(['rand_resp','cmltv_p','cmltv_n',], axis=1)

        print('average response rate: ' , avg_resp_rate)
        return crt    
  
    
    """
    Método que apresenta a tabela de confusão com multiplos bins.
    @param test Label de teste
    @param pred Label predita
    @param bins Bins a serem considerados para a matriz de confusão
    @return pandas.DataFrame Retorna dataframe com a avaliação do modelo
    """  
    @staticmethod
    def confusion_matrix (test, pred, bins):

        res = pd.DataFrame(np.column_stack((test, pred)),
                           columns=['Target','PR_0'])

        
        res['pred_grp'] = pd.cut(res['PR_0'].rank(method='first'), bins, labels=False)
        res['target_grp'] = pd.cut(res['Target'].rank(method='first'), bins, labels=False)
        
        ret = res.drop(['Target','PR_0'],axis=1)
        ret['cnt'] = 1
        ret = ret.groupby(['pred_grp','target_grp']).agg({'cnt':'sum'})
        
        return ret.reset_index()
