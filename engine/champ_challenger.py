

class ChampChallenger():
    
    def __init__(self,model_name,model_type,safra):
        
        query = """SELECT * FROM ML_TBL.TB_CHAMP_CHALLENGER
                    WHERE SAFRA <= {safra}
                    AND MODEL_NAME = {model_name}
                    AND MODEL_TYPE = {model_type}
                    AND DT_SCORE = (
                                    SELECT MAX(DT_SCORE) FROM ML_TBL.TB_CHAMP_CHALLENGER
                                    WHERE SAFRA <= {safra}
                                    AND MODEL_NAME = {model_name}
                                    AND MODEL_TYPE = {model_type}
                                    )
                """.format(safra=safra,model_name=model_name,model_type=model_type)
        
        self.data = bigquery.execute_response(query,output='df')
        
    def compare(self,df,comp_func):