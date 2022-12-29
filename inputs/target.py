import pandas as pd
import numpy as np
from melitk.connectors import BigQuery
import json
import base64
import os
from datetime import datetime

def gen_target(safra):
    # Connecting to BQ
    credentials = json.loads(base64.b64decode(os.environ["SECRET_DS_GPC_KEY"]))
    bigquery = BigQuery(
        credentials=credentials
    )    
    query = """SELECT CUS_CUST_ID,
                        CASE WHEN COUNT(*) < 10 THEN count(*)
                        ELSE 10
                        END target
                WHERE count(*) > 0
            FROM (
                SELECT DISTINCT CUS_CUST_ID,CRD_CRED_CARD_UUID
                FROM  mp-open-finance.BI_OPEN_FINANCE.LK_MP_OPF_DATA_IN_CRED_CARD_ACC_SCDT2
                WHERE CRD_CRED_CARD_UUID IS NOT NULL
            )
            GROUP BY 1
            """
    
    df = bigquery.execute_response(query,output='df')
    
    return df

