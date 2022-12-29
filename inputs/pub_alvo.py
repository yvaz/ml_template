import sklearn.datasets
import pandas as pd
import numpy as np
from melitk.connectors import BigQuery
import json
import base64
import os
from datetime import datetime

def gen_pub(safra):

    # Connecting to BQ
    credentials = json.loads(base64.b64decode(os.environ["SECRET_DS_GPC_KEY"]))
    bigquery = BigQuery(
        credentials=credentials
    )

    safra_ini = datetime.strftime(
                    datetime.strptime(safra,"%Y%m")
                ,"%Y-%m-01")
    
    query = """SELECT CUS_CUST_ID
                WHERE count(*) > 0
            FROM (
                SELECT DISTINCT CUS_CUST_ID,CRD_CRED_CARD_UUID
                FROM  mp-open-finance.BI_OPEN_FINANCE.LK_MP_OPF_DATA_IN_CRED_CARD_ACC_SCDT2
                WHERE CRD_CRED_CARD_UUID IS NOT NULL
            )
            GROUP BY 1""".format(safra=safra_ini)

    df = bigquery.execute_response(query,  output="df")
    return df

