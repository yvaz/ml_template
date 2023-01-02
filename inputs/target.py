import sklearn.datasets
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

    safra_ini = datetime.strftime(
                    datetime.strptime(safra,"%Y%m")
                ,"%Y-%m-%d")
    
    query = """
                WITH CUSTS_IN AS (
                    SELECT
                        cus_cust_id,
                        CASE
                            WHEN CAST(CNS_CONSENT_STATUS_UPD_DTTM AS DATE) >= DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 1 MONTH), MONTH)
                            AND CAST(CNS_CONSENT_STATUS_UPD_DTTM AS DATE) <= LAST_DAY(
                                DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 1 MONTH), MONTH),
                                MONTH
                            ) THEN 'S'
                            ELSE 'N'
                        END AS FLAG_IN_MES
                    FROM
                        mp-open-finance.BI_OPEN_FINANCE.LK_MP_OPF_DATA_IN_DATA_OUT_CONSENT_SCDT2
                    WHERE
                        cus_cust_id IS NOT NULL
                        AND CNS_CONSENT_TYPE_DETAIL = 'DATA_IN'
                        AND CNS_CONSENT_STATUS_DETAIL IN (
                            'AUTHORISED',
                            'AWAITING_AUTHORISATION',
                            'REJECTED_BY_SYSTEM',
                            'RENEWED'
                        )
                        AND CNS_CONSENT_STATUS_UPD_DTTM < LAST_DAY(
                            DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 1 MONTH), MONTH),
                            MONTH
                        ) QUALIFY row_number() OVER (
                            PARTITION by CUS_CUST_ID
                            ORDER BY
                                CNS_CONSENT_STATUS_UPD_DTTM DESC
                        ) = 1
                ),
                CUST_ATIVO AS (
                    SELECT
                        DISTINCT cus_cust_id,
                        marketplace,
                        STATUS
                    FROM
                        WHOWNER.APP_DEVICES
                    WHERE
                        marketplace IN ('MERCADOPAGO')
                        AND STATUS IN ('ACTIVE')
                        AND active = 'YES'
                        AND platform IN ('ios', 'android')
                        AND sit_site_id = 'MLB'
                        AND token IS NOT NULL
                        AND TRIM(TOKEN) <> ''
                ),
                EMPLOYERS_MELI AS (
                    SELECT
                        DISTINCT cus_cust_id
                    FROM
                        WHOWNER.LK_MELI_EMPLOYEES
                    WHERE
                        UPPER(STATUS) = 'ACTIVO'
                        AND END_DATE IS NULL
                ),
                CLUSTER_CUST AS (
                    SELECT
                        CUS_CUST_ID,
                        SEGMENT,
                        TIM_MONTH_ID,
                        ROW_NUMBER() OVER (
                            PARTITION BY CUS_CUST_ID
                            ORDER BY
                                TIM_MONTH_ID DESC
                        ) AS RANKING
                    FROM
                        WHOWNER.LK_MP_MAUS_CHARACTERS
                    WHERE
                        SIT_SITE_ID = 'MLB' QUALIFY RANKING = 1
                ),
                PUSH30 AS (
                SELECT
                    CUS_CUST_ID,
                    SUM(
                        CASE
                            WHEN NOT_NOTI_EVENT_TYPE_ID = 'shown' THEN 1
                            ELSE 0
                        END
                    ) AS NR_SHOWNS,
                    SUM(
                        CASE
                            WHEN NOT_NOTI_EVENT_TYPE_ID = 'open' THEN 1
                            ELSE 0
                        END
                    ) AS NR_OPENS,
                    MIN(
                        CASE
                            WHEN NOT_NOTI_EVENT_TYPE_ID = 'shown' THEN NOT_NOTI_DS_DT
                            ELSE NULL
                        END
                    ) AS MIN_NOT_DATE_SHOWN,
                    MIN(
                        CASE
                            WHEN NOT_NOTI_EVENT_TYPE_ID = 'open' THEN NOT_NOTI_DS_DT
                            ELSE NULL
                        END
                    ) AS MIN_NOT_DATE_OPEN
                FROM
                    WHOWNER.BT_NOTIFICATION_CAMPAIGN
                WHERE
                    SIT_SITE_ID = 'MLB'
                    AND NOT_NOTI_PATH = 'NOTIFICATION_CAMPAIGN'
                    AND NOT_NOTI_BUSINESS = 'mercadopago'
                    AND NOT_NOTI_DS_DT BETWEEN DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 1 MONTH), MONTH)
                    AND LAST_DAY(
                        DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 1 MONTH), MONTH),
                        MONTH
                    )
                    AND NOT_NOTI_EVENT_TYPE_ID IN ('open', 'shown')
                    AND NOT_NOTI_BATCH_ID NOT LIKE '%BLACKLIST%'
                GROUP BY
                    1
                ),
                SALDO_CONTA30 AS (
                    SELECT
                        CUS_CUST_ID,
                        MAX(AVAILABLE_BALANCE) AS AVAILABLE_BALANCE
                    FROM
                        WHOWNER.BT_MP_SALDOS_SITE
                    WHERE
                        TIM_DAY BETWEEN DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 1 MONTH), MONTH)
                        AND LAST_DAY(
                            DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 1 MONTH), MONTH),
                            MONTH
                        )
                    GROUP BY
                        CUS_CUST_ID
                )
                SELECT
                    CA.CUS_CUST_ID AS CUS_CUST_ID, 1 as target
                FROM
                    CUST_ATIVO AS CA
                    LEFT JOIN PUSH30 AS P30 ON CA.cus_cust_id = P30.cus_cust_id
                    LEFT JOIN SALDO_CONTA30 AS SC30 ON CA.cus_cust_id = SC30.cus_cust_id
                    LEFT JOIN CLUSTER_CUST AS CC ON CA.cus_cust_id = CC.cus_cust_id
                    LEFT JOIN EMPLOYERS_MELI AS EM ON CA.cus_cust_id = EM.cus_cust_id
                    LEFT JOIN CUSTS_IN AS CI ON CA.cus_cust_id = CI.CUS_CUST_ID
                WHERE
                    STATUS = 'ACTIVE'
                    -- AND SC30.AVAILABLE_BALANCE > 0
                    AND EM.cus_cust_id IS NULL
                    AND P30.CUS_CUST_ID IS NOT NULL
                    -- AND P30.NR_OPENS > 0
                    AND CC.SEGMENT = 'SELLER'
                    AND CI.FLAG_IN_MES = 'S'""".format(safra=safra_ini)

    df = bigquery.execute_response(query,  output="df")
    return df
