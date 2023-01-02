import sklearn.datasets
import pandas as pd
import numpy as np
from melitk.connectors import BigQuery
import json
import base64
import os
from datetime import datetime
import yaml
from yaml.loader import SafeLoader

def gen_scores(safra):

# Connecting to BQ
    credentials = json.loads(base64.b64decode(os.environ["SECRET_DS_GPC_KEY"]))
    bigquery = BigQuery(
        credentials=credentials
    )

    safra_ini = datetime.strftime(
                    datetime.strptime(safra,"%Y%m")
                ,"%Y-%m-%d")
    
    with open(os.path.dirname(__file__)+'/../engine/main_cfg.yaml','r') as fp:
        config = yaml.load(fp, Loader = SafeLoader)
        
    model_name = config['model_name']
    
    query = """
        WITH CUSTS_OUT AS (
        SELECT
            DISTINCT cus_cust_id
        FROM
            mp-open-finance.BI_OPEN_FINANCE.LK_MP_OPF_DATA_IN_DATA_OUT_CONSENT_SCDT2
        WHERE
            cus_cust_id IS NOT NULL
            AND CNS_CONSENT_TYPE_DETAIL = 'DATA_OUT'
        ),
        CUSTS_IN AS (
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
                WHOWNER.LK_MP_OPF_DATA_IN_DATA_OUT_CONSENT_SCDT2
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
        PUSH_OPF AS (
            SELECT
                CUS_CUST_ID,
                SUM(
                    CASE
                        WHEN NOT_NOTI_EVENT_TYPE_ID = 'shown' THEN 1
                        ELSE 0
                    END
                ) AS NR_SHOWNS_OPF,
                SUM(
                    CASE
                        WHEN NOT_NOTI_EVENT_TYPE_ID = 'open' THEN 1
                        ELSE 0
                    END
                ) AS NR_OPENS_OPF,
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
                AND NOT_NOTI_DS_DT < LAST_DAY(
                    DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 1 MONTH), MONTH),
                    MONTH
                )
                AND NOT_NOTI_EVENT_TYPE_ID IN ('open', 'shown')
                AND NOT_NOTI_BATCH_ID NOT LIKE '%BLACKLIST%'
                AND (
                    NOT_NOTI_CAMPAIGN_ID LIKE '%OPEN-FINANCE%'
                    OR NOT_NOTI_CAMPAIGN_ID LIKE '%OPF%'
                )
            GROUP BY
                1
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
        PUSH60 AS (
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
                AND NOT_NOTI_DS_DT BETWEEN DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 2 MONTH), MONTH)
                AND LAST_DAY(
                    DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 2 MONTH), MONTH),
                    MONTH
                )
                AND NOT_NOTI_EVENT_TYPE_ID IN ('open', 'shown')
                AND NOT_NOTI_BATCH_ID NOT LIKE '%BLACKLIST%'
            GROUP BY
                1
        ),
        PUSH90 AS (
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
                AND NOT_NOTI_DS_DT BETWEEN DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 3 MONTH), MONTH)
                AND LAST_DAY(
                    DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 3 MONTH), MONTH),
                    MONTH
                )
                AND NOT_NOTI_EVENT_TYPE_ID IN ('open', 'shown')
                AND NOT_NOTI_BATCH_ID NOT LIKE '%CG'
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
        ),
        SALDO_CONTA60 AS (
            SELECT
                CUS_CUST_ID,
                MAX(AVAILABLE_BALANCE) AS AVAILABLE_BALANCE
            FROM
                WHOWNER.BT_MP_SALDOS_SITE
            WHERE
                TIM_DAY BETWEEN DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 2 MONTH), MONTH)
                AND LAST_DAY(
                    DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 2 MONTH), MONTH),
                    MONTH
                )
            GROUP BY
                CUS_CUST_ID
        ),
        SALDO_CONTA90 AS (
            SELECT
                CUS_CUST_ID,
                MAX(AVAILABLE_BALANCE) AS AVAILABLE_BALANCE
            FROM
                WHOWNER.BT_MP_SALDOS_SITE
            WHERE
                TIM_DAY BETWEEN DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 3 MONTH), MONTH)
                AND LAST_DAY(
                    DATE_TRUNC(DATE_SUB('{safra}', INTERVAL 3 MONTH), MONTH),
                    MONTH
                )
            GROUP BY
                CUS_CUST_ID
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
        EMPLOYERS_MELI AS (
            SELECT
                DISTINCT cus_cust_id
            FROM
                WHOWNER.LK_MELI_EMPLOYEES
            WHERE
                UPPER(STATUS) = 'ACTIVO'
                AND END_DATE IS NULL
        ),
        PRINCIPALIDADE AS (
            SELECT
                CUS_CUST_ID,
                STAGE_LIFECYCLE,
                CATEGORY_PRINC
            FROM
                WHOWNER.LK_MP_MAUS_LIFECYCLE
            WHERE
                SIT_SITE_ID = 'MLB' QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY CUS_CUST_ID
                    ORDER BY
                        PHOTO_ID DESC
                ) = 1
        ),
        MAU30 AS (
            SELECT
                CUS_CUST_ID,
                SUM(
                    MAS_POINT_TPV_AMT + MAS_QR_TPV_AMT + MAS_OP_TPV_AMT + MAS_ON_TPV_AMT
                ) AS TPV_SELLER,
                SUM(
                    MAF_TED_TPV_AMT + MAF_PIX_TPV_AMT + MAF_DEBCAIXA_TPV_AMT + MAF_POR_TPV_AMT + MAF_PEC_TPV_AMT + MAF_CLABE_TPV_AMT + MAF_TDEBITO_TPV_AMT + MAF_EFECTIVO_TPV_AMT + MAF_DEBIN_TPV_AMT + MAF_CASHIN_TPV_AMT + MAF_LOTERIA_TPV
                ) AS TPV_FUNDER,
                SUM(
                    MAR_TED_TPV_AMT + MAR_PIX_TPV_AMT + MAR_P2P_TPV_AMT + MAR_BPS_TPV_AMT + MAR_REFUND_TPV_AMT + MAR_REMESSAS_TPV + MAR_CLABE_TPV + MAR_CASHBACK_TPV
                ) AS TPV_RECIEVER,
                SUM(MAB_CSR_CRD_AMT + MAB_MRC_CRD_AMT) AS TPV_BORROWER,
                SUM(
                    MAP1_QR_TPV_AMT + MAP1_CC_TPV_AMT + MAP1_DC_TPV_AMT + MAP1_ACQUR_TPV + MAP1_QRPIX_TPV + MAP1_QR_NO_MP_TPV_AMT
                ) AS TPV_COMPRAS_FISICAS,
                SUM(
                    MAP2_CHO_TPV_AMT + MAP2_AMML_TPV_AMT + MAP2_CRD_TPV_AMT + MAP2_TVC_TPV_AMT + MAP2_TVD_TPV_AMT
                ) AS TPV_COMPRAS_WEB,
                SUM(
                    MAP3_BP_TPV_AMT + MAP3_RCH_TPV_AMT + MAP3_SIG_TPV_AMT + MAP3_DON_TPV_AMT + MAP3_TRANSP_TPV_AMT + MAP3_PASE_TPV_AMT
                ) AS TPV_COMPRAS_APP,
                SUM(MAP4_INSR_TPV_AMT + MAP4_CRIPTO_TPV_AMT) AS TPV_PRODUTOS_BANCARIOS,
                SUM(
                    MAP5_TED_WTHDRWL_AMT + MAP5_PIX_WTHDRWL_AMT + MAP5_P2P_WTHDRWL_AMT + MAP5_CBU_CVU_AMT + MAP5_CLABEON_AMT + MAP5_CLABEOFF_AMT
                ) AS TPV_TRANSFERENCIAS_DIFFTIT,
                SUM(
                    MAW1_WIT_WTHDRWL_AMT + MAW1_TED_WTHDRWL_AMT + MAW1_PIX_WTHDRWL_AMT + MAW1_CBU_CVU_AMT + MAW1_CLABEOFF_AMT + MAW1_CLABEON_AMT
                ) AS TPV_TRANSFERENCIAS_MSMTIT,
                SUM(
                    MAW2_CARD_WTHDRWL_AMT + MAW2_QR_WTHDRWL_AMT + MAW2_EXTRACASH_AMT + MAW2_RAPIPAGO_AMT
                ) AS TPV_SAQUES,
                SUM(
                    MAS_POINT_TPN + MAS_QR_TPN + MAS_OP_TPN + MAS_ON_TPN
                ) AS TPN_SELLER,
                SUM(
                    MAF_TED_TPN + MAF_PIX_TPN + MAF_DEBCAIXA_TPN + MAF_POR_TPN + MAF_PEC_TPN + MAF_CLABE_TPN + MAF_TDEBITO_TPN + MAF_EFECTIVO_TPN + MAF_DEBIN_TPN + MAF_CBU_CVU_TPN + MAF_DEBITO_PEI_TPN + MAF_DEBITO_NOPEI_TPN + MAF_CASHIN_TPN + MAF_LOTERIA_TPN
                ) AS TPN_FUNDER,
                SUM(
                    MAR_TED_TPN + MAR_PIX_TPN + MAR_P2P_TPN + MAR_BPS_TPN + MAR_REFUND_TPN + MAR_REMESSAS_TPN + MAR_CLABE_TPN + MAR_CBU_CVU_TPN + MAR_CASHBACK_TPN
                ) AS TPN_RECIEVER,
                SUM(MAB_CSR_CRD_TPN + MAB_MRC_CRD_TPN) AS TPN_BORROWER,
                SUM(
                    MAP1_QR_TPN + MAP1_CC_TPN + MAP1_DC_TPN + MAP1_ACQUR_TPN + MAP1_QRPIX_TPN + MAP1_QR_NO_MP_TPN
                ) AS TPN_COMPRAS_FISICAS,
                SUM(
                    MAP2_CHO_TPN + MAP2_AMML_TPN + MAP2_CRD_TPN + MAP2_TVC_TPN + MAP2_TVD_TPN
                ) AS TPN_COMPRAS_WEB,
                SUM(
                    MAP3_BP_TPN + MAP3_RCH_TPN + MAP3_SIG_TPN + MAP3_DON_TPN + MAP3_TRANSP_TPN + MAP3_PASE_TPN + MAP3_DIG_GOODS_TPN + MAP3_DELIVERY_TPN
                ) AS TPN_COMPRAS_APP,
                SUM(MAP4_INSR_TPN + MAP4_CRIPTO_TPN) AS TPN_PRODUTOS_BANCARIOS,
                SUM(
                    MAP5_TED_WTHDRWL_TPN + MAP5_PIX_WTHDRWL_TPN + MAP5_P2P_WTHDRWL_TPN + MAP5_CBU_CVU_TPN + MAP5_CLABEON_TPN + MAP5_CLABEOFF_TPN
                ) AS TPN_TRANSFERENCIAS_DIFFTIT,
                SUM(
                    MAW1_WIT_WTHDRWL_TPN + MAW1_TED_WTHDRWL_TPN + MAW1_PIX_WTHDRWL_TPN + MAW1_CBU_CVU_TPN + MAW1_CLABEOFF_TPN + MAW1_CLABEON_TPN
                ) AS TPN_TRANSFERENCIAS_MSMTIT,
                SUM(
                    MAW2_CARD_WTHDRWL_TPN + MAW2_QR_WTHDRWL_TPN + MAW2_EXTRACASH_TPN + MAW2_RAPIPAGO_TPN
                ) AS TPN_SAQUES
            FROM
                WHOWNER.BT_MP_MAUS_DETAIL
            WHERE
                SIT_SITE_ID = 'MLB'
                AND TIM_MONTH_ID = CAST(
                    FORMAT_DATE("%Y%m", DATE_SUB('{safra}', INTERVAL 1 MONTH)) AS INT64
                )
            GROUP BY
                1
        ),
        MAU60 AS (
            SELECT
                CUS_CUST_ID,
                SUM(
                    MAS_POINT_TPV_AMT + MAS_QR_TPV_AMT + MAS_OP_TPV_AMT + MAS_ON_TPV_AMT
                ) AS TPV_SELLER,
                SUM(
                    MAF_TED_TPV_AMT + MAF_PIX_TPV_AMT + MAF_DEBCAIXA_TPV_AMT + MAF_POR_TPV_AMT + MAF_PEC_TPV_AMT + MAF_CLABE_TPV_AMT + MAF_TDEBITO_TPV_AMT + MAF_EFECTIVO_TPV_AMT + MAF_DEBIN_TPV_AMT + MAF_CASHIN_TPV_AMT + MAF_LOTERIA_TPV
                ) AS TPV_FUNDER,
                SUM(
                    MAR_TED_TPV_AMT + MAR_PIX_TPV_AMT + MAR_P2P_TPV_AMT + MAR_BPS_TPV_AMT + MAR_REFUND_TPV_AMT + MAR_REMESSAS_TPV + MAR_CLABE_TPV + MAR_CASHBACK_TPV
                ) AS TPV_RECIEVER,
                SUM(MAB_CSR_CRD_AMT + MAB_MRC_CRD_AMT) AS TPV_BORROWER,
                SUM(
                    MAP1_QR_TPV_AMT + MAP1_CC_TPV_AMT + MAP1_DC_TPV_AMT + MAP1_ACQUR_TPV + MAP1_QRPIX_TPV + MAP1_QR_NO_MP_TPV_AMT
                ) AS TPV_COMPRAS_FISICAS,
                SUM(
                    MAP2_CHO_TPV_AMT + MAP2_AMML_TPV_AMT + MAP2_CRD_TPV_AMT + MAP2_TVC_TPV_AMT + MAP2_TVD_TPV_AMT
                ) AS TPV_COMPRAS_WEB,
                SUM(
                    MAP3_BP_TPV_AMT + MAP3_RCH_TPV_AMT + MAP3_SIG_TPV_AMT + MAP3_DON_TPV_AMT + MAP3_TRANSP_TPV_AMT + MAP3_PASE_TPV_AMT
                ) AS TPV_COMPRAS_APP,
                SUM(MAP4_INSR_TPV_AMT + MAP4_CRIPTO_TPV_AMT) AS TPV_PRODUTOS_BANCARIOS,
                SUM(
                    MAP5_TED_WTHDRWL_AMT + MAP5_PIX_WTHDRWL_AMT + MAP5_P2P_WTHDRWL_AMT + MAP5_CBU_CVU_AMT + MAP5_CLABEON_AMT + MAP5_CLABEOFF_AMT
                ) AS TPV_TRANSFERENCIAS_DIFFTIT,
                SUM(
                    MAW1_WIT_WTHDRWL_AMT + MAW1_TED_WTHDRWL_AMT + MAW1_PIX_WTHDRWL_AMT + MAW1_CBU_CVU_AMT + MAW1_CLABEOFF_AMT + MAW1_CLABEON_AMT
                ) AS TPV_TRANSFERENCIAS_MSMTIT,
                SUM(
                    MAW2_CARD_WTHDRWL_AMT + MAW2_QR_WTHDRWL_AMT + MAW2_EXTRACASH_AMT + MAW2_RAPIPAGO_AMT
                ) AS TPV_SAQUES,
                SUM(
                    MAS_POINT_TPN + MAS_QR_TPN + MAS_OP_TPN + MAS_ON_TPN
                ) AS TPN_SELLER,
                SUM(
                    MAF_TED_TPN + MAF_PIX_TPN + MAF_DEBCAIXA_TPN + MAF_POR_TPN + MAF_PEC_TPN + MAF_CLABE_TPN + MAF_TDEBITO_TPN + MAF_EFECTIVO_TPN + MAF_DEBIN_TPN + MAF_CBU_CVU_TPN + MAF_DEBITO_PEI_TPN + MAF_DEBITO_NOPEI_TPN + MAF_CASHIN_TPN + MAF_LOTERIA_TPN
                ) AS TPN_FUNDER,
                SUM(
                    MAR_TED_TPN + MAR_PIX_TPN + MAR_P2P_TPN + MAR_BPS_TPN + MAR_REFUND_TPN + MAR_REMESSAS_TPN + MAR_CLABE_TPN + MAR_CBU_CVU_TPN + MAR_CASHBACK_TPN
                ) AS TPN_RECIEVER,
                SUM(MAB_CSR_CRD_TPN + MAB_MRC_CRD_TPN) AS TPN_BORROWER,
                SUM(
                    MAP1_QR_TPN + MAP1_CC_TPN + MAP1_DC_TPN + MAP1_ACQUR_TPN + MAP1_QRPIX_TPN + MAP1_QR_NO_MP_TPN
                ) AS TPN_COMPRAS_FISICAS,
                SUM(
                    MAP2_CHO_TPN + MAP2_AMML_TPN + MAP2_CRD_TPN + MAP2_TVC_TPN + MAP2_TVD_TPN
                ) AS TPN_COMPRAS_WEB,
                SUM(
                    MAP3_BP_TPN + MAP3_RCH_TPN + MAP3_SIG_TPN + MAP3_DON_TPN + MAP3_TRANSP_TPN + MAP3_PASE_TPN + MAP3_DIG_GOODS_TPN + MAP3_DELIVERY_TPN
                ) AS TPN_COMPRAS_APP,
                SUM(MAP4_INSR_TPN + MAP4_CRIPTO_TPN) AS TPN_PRODUTOS_BANCARIOS,
                SUM(
                    MAP5_TED_WTHDRWL_TPN + MAP5_PIX_WTHDRWL_TPN + MAP5_P2P_WTHDRWL_TPN + MAP5_CBU_CVU_TPN + MAP5_CLABEON_TPN + MAP5_CLABEOFF_TPN
                ) AS TPN_TRANSFERENCIAS_DIFFTIT,
                SUM(
                    MAW1_WIT_WTHDRWL_TPN + MAW1_TED_WTHDRWL_TPN + MAW1_PIX_WTHDRWL_TPN + MAW1_CBU_CVU_TPN + MAW1_CLABEOFF_TPN + MAW1_CLABEON_TPN
                ) AS TPN_TRANSFERENCIAS_MSMTIT,
                SUM(
                    MAW2_CARD_WTHDRWL_TPN + MAW2_QR_WTHDRWL_TPN + MAW2_EXTRACASH_TPN + MAW2_RAPIPAGO_TPN
                ) AS TPN_SAQUES
            FROM
                WHOWNER.BT_MP_MAUS_DETAIL
            WHERE
                SIT_SITE_ID = 'MLB'
                AND TIM_MONTH_ID = CAST(
                    FORMAT_DATE("%Y%m", DATE_SUB('{safra}', INTERVAL 2 MONTH)) AS INT64
                )
            GROUP BY
                1
        ),
        MAU90 AS (
            SELECT
                CUS_CUST_ID,
                SUM(
                    MAS_POINT_TPV_AMT + MAS_QR_TPV_AMT + MAS_OP_TPV_AMT + MAS_ON_TPV_AMT
                ) AS TPV_SELLER,
                SUM(
                    MAF_TED_TPV_AMT + MAF_PIX_TPV_AMT + MAF_DEBCAIXA_TPV_AMT + MAF_POR_TPV_AMT + MAF_PEC_TPV_AMT + MAF_CLABE_TPV_AMT + MAF_TDEBITO_TPV_AMT + MAF_EFECTIVO_TPV_AMT + MAF_DEBIN_TPV_AMT + MAF_CASHIN_TPV_AMT + MAF_LOTERIA_TPV
                ) AS TPV_FUNDER,
                SUM(
                    MAR_TED_TPV_AMT + MAR_PIX_TPV_AMT + MAR_P2P_TPV_AMT + MAR_BPS_TPV_AMT + MAR_REFUND_TPV_AMT + MAR_REMESSAS_TPV + MAR_CLABE_TPV + MAR_CASHBACK_TPV
                ) AS TPV_RECIEVER,
                SUM(MAB_CSR_CRD_AMT + MAB_MRC_CRD_AMT) AS TPV_BORROWER,
                SUM(
                    MAP1_QR_TPV_AMT + MAP1_CC_TPV_AMT + MAP1_DC_TPV_AMT + MAP1_ACQUR_TPV + MAP1_QRPIX_TPV + MAP1_QR_NO_MP_TPV_AMT
                ) AS TPV_COMPRAS_FISICAS,
                SUM(
                    MAP2_CHO_TPV_AMT + MAP2_AMML_TPV_AMT + MAP2_CRD_TPV_AMT + MAP2_TVC_TPV_AMT + MAP2_TVD_TPV_AMT
                ) AS TPV_COMPRAS_WEB,
                SUM(
                    MAP3_BP_TPV_AMT + MAP3_RCH_TPV_AMT + MAP3_SIG_TPV_AMT + MAP3_DON_TPV_AMT + MAP3_TRANSP_TPV_AMT + MAP3_PASE_TPV_AMT
                ) AS TPV_COMPRAS_APP,
                SUM(MAP4_INSR_TPV_AMT + MAP4_CRIPTO_TPV_AMT) AS TPV_PRODUTOS_BANCARIOS,
                SUM(
                    MAP5_TED_WTHDRWL_AMT + MAP5_PIX_WTHDRWL_AMT + MAP5_P2P_WTHDRWL_AMT + MAP5_CBU_CVU_AMT + MAP5_CLABEON_AMT + MAP5_CLABEOFF_AMT
                ) AS TPV_TRANSFERENCIAS_DIFFTIT,
                SUM(
                    MAW1_WIT_WTHDRWL_AMT + MAW1_TED_WTHDRWL_AMT + MAW1_PIX_WTHDRWL_AMT + MAW1_CBU_CVU_AMT + MAW1_CLABEOFF_AMT + MAW1_CLABEON_AMT
                ) AS TPV_TRANSFERENCIAS_MSMTIT,
                SUM(
                    MAW2_CARD_WTHDRWL_AMT + MAW2_QR_WTHDRWL_AMT + MAW2_EXTRACASH_AMT + MAW2_RAPIPAGO_AMT
                ) AS TPV_SAQUES,
                SUM(
                    MAS_POINT_TPN + MAS_QR_TPN + MAS_OP_TPN + MAS_ON_TPN
                ) AS TPN_SELLER,
                SUM(
                    MAF_TED_TPN + MAF_PIX_TPN + MAF_DEBCAIXA_TPN + MAF_POR_TPN + MAF_PEC_TPN + MAF_CLABE_TPN + MAF_TDEBITO_TPN + MAF_EFECTIVO_TPN + MAF_DEBIN_TPN + MAF_CBU_CVU_TPN + MAF_DEBITO_PEI_TPN + MAF_DEBITO_NOPEI_TPN + MAF_CASHIN_TPN + MAF_LOTERIA_TPN
                ) AS TPN_FUNDER,
                SUM(
                    MAR_TED_TPN + MAR_PIX_TPN + MAR_P2P_TPN + MAR_BPS_TPN + MAR_REFUND_TPN + MAR_REMESSAS_TPN + MAR_CLABE_TPN + MAR_CBU_CVU_TPN + MAR_CASHBACK_TPN
                ) AS TPN_RECIEVER,
                SUM(MAB_CSR_CRD_TPN + MAB_MRC_CRD_TPN) AS TPN_BORROWER,
                SUM(
                    MAP1_QR_TPN + MAP1_CC_TPN + MAP1_DC_TPN + MAP1_ACQUR_TPN + MAP1_QRPIX_TPN + MAP1_QR_NO_MP_TPN
                ) AS TPN_COMPRAS_FISICAS,
                SUM(
                    MAP2_CHO_TPN + MAP2_AMML_TPN + MAP2_CRD_TPN + MAP2_TVC_TPN + MAP2_TVD_TPN
                ) AS TPN_COMPRAS_WEB,
                SUM(
                    MAP3_BP_TPN + MAP3_RCH_TPN + MAP3_SIG_TPN + MAP3_DON_TPN + MAP3_TRANSP_TPN + MAP3_PASE_TPN + MAP3_DIG_GOODS_TPN + MAP3_DELIVERY_TPN
                ) AS TPN_COMPRAS_APP,
                SUM(MAP4_INSR_TPN + MAP4_CRIPTO_TPN) AS TPN_PRODUTOS_BANCARIOS,
                SUM(
                    MAP5_TED_WTHDRWL_TPN + MAP5_PIX_WTHDRWL_TPN + MAP5_P2P_WTHDRWL_TPN + MAP5_CBU_CVU_TPN + MAP5_CLABEON_TPN + MAP5_CLABEOFF_TPN
                ) AS TPN_TRANSFERENCIAS_DIFFTIT,
                SUM(
                    MAW1_WIT_WTHDRWL_TPN + MAW1_TED_WTHDRWL_TPN + MAW1_PIX_WTHDRWL_TPN + MAW1_CBU_CVU_TPN + MAW1_CLABEOFF_TPN + MAW1_CLABEON_TPN
                ) AS TPN_TRANSFERENCIAS_MSMTIT,
                SUM(
                    MAW2_CARD_WTHDRWL_TPN + MAW2_QR_WTHDRWL_TPN + MAW2_EXTRACASH_TPN + MAW2_RAPIPAGO_TPN
                ) AS TPN_SAQUES
            FROM
                WHOWNER.BT_MP_MAUS_DETAIL
            WHERE
                SIT_SITE_ID = 'MLB'
                AND TIM_MONTH_ID = CAST(
                    FORMAT_DATE("%Y%m", DATE_SUB('{safra}', INTERVAL 3 MONTH)) AS INT64
                )
            GROUP BY
                1
        ),
        CADASTRO AS (
            SELECT
                CUS_CUST_ID,
                CUS_RU_SINCE_DT
            FROM
                `meli-bi-data.WHOWNER.LK_CUS_CUSTOMERS_DATA`
        ),
        CARTAOCREDITO AS (
            SELECT
                DISTINCT CUS_CUST_ID
            FROM
                WHOWNER.BT_CCARD_ACCOUNT
            WHERE
                UPPER(CCARD_ACCOUNT_STATUS) = 'ACTIVE'
        ),
        LOYALTY AS (
            SELECT
                CUS_CUST_ID,
                LYL_LEVEL_NUMBER
            FROM
                `meli-bi-data.WHOWNER.LK_LYL_USER_LEVEL_LOG`
            WHERE
                SIT_SITE_ID = 'MLB' QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY CUS_CUST_ID
                    ORDER BY
                        LYL_LEVEL_FROM DESC
                ) = 1
        ),
        GEOLOC AS (
            SELECT
                CUS_CUST_ID,
                GEO_LAT,
                GEO_LONG
            FROM
                `meli-bi-data.WHOWNER.LK_MP_GEODATA_MAU`
            WHERE
                SIT_SITE_ID = 'MLB' qualify row_number() over (
                    PARTITION by CUS_CUST_ID
                    ORDER BY
                        TIM_MONTH_ID DESC
                ) = 1
        )
        SELECT
            CA.CUS_CUST_ID AS CUS_CUST_ID,
            CC.SEGMENT AS CLUSTER,
            STAGE_LIFECYCLE AS CICLO_DE_VIDA,
            CATEGORY_PRINC AS PRINCIPALIDADE,
            CAST(LYL.LYL_LEVEL_NUMBER AS INT64) AS LOYALTY_LVL,
            CASE
                WHEN CO.cus_cust_id IS NULL THEN 'N'
                ELSE 'S'
            END AS FLAG_DATA_OUT,
            CASE
                WHEN CART.CUS_CUST_ID IS NULL THEN 'N'
                ELSE 'S'
            END AS FLAG_CARTAO_CREDITO,
            SC30.AVAILABLE_BALANCE AS SALDO_EM_CONTA30,
            SC60.AVAILABLE_BALANCE AS SALDO_EM_CONTA60,
            SC90.AVAILABLE_BALANCE AS SALDO_EM_CONTA90,
            CAST(M30.TPN_SELLER AS INT64) AS TPN_SELLER30,
            CAST(M30.TPN_FUNDER AS INT64) AS TPN_FUNDER30,
            CAST(M30.TPN_RECIEVER AS INT64) AS TPN_RECIEVER30,
            CAST(M30.TPN_BORROWER AS INT64) AS TPN_BORROWER30,
            CAST(M30.TPN_COMPRAS_WEB AS INT64) AS TPN_COMPRAS_WEB30,
            CAST(M30.TPN_COMPRAS_APP AS INT64) AS TPN_COMPRAS_APP30,
            CAST(M30.TPN_PRODUTOS_BANCARIOS AS INT64) AS TPN_PRODUTOS_BANCARIOS30,
            CAST(M30.TPN_TRANSFERENCIAS_DIFFTIT AS INT64) AS TPN_TRANSFERENCIAS_DIFFTIT30,
            CAST(M30.TPN_TRANSFERENCIAS_MSMTIT AS INT64) AS TPN_TRANSFERENCIAS_MSMTIT30,
            CAST(M30.TPN_SAQUES AS INT64) AS TPN_SAQUES30,
            CAST(M60.TPN_SELLER AS INT64) AS TPN_SELLER60,
            CAST(M60.TPN_FUNDER AS INT64) AS TPN_FUNDER60,
            CAST(M60.TPN_RECIEVER AS INT64) AS TPN_RECIEVER60,
            CAST(M60.TPN_BORROWER AS INT64) AS TPN_BORROWER60,
            CAST(M60.TPN_COMPRAS_WEB AS INT64) AS TPN_COMPRAS_WEB60,
            CAST(M60.TPN_COMPRAS_APP AS INT64) AS TPN_COMPRAS_APP60,
            CAST(M60.TPN_PRODUTOS_BANCARIOS AS INT64) AS TPN_PRODUTOS_BANCARIOS60,
            CAST(M60.TPN_TRANSFERENCIAS_DIFFTIT AS INT64) AS TPN_TRANSFERENCIAS_DIFFTIT60,
            CAST(M60.TPN_TRANSFERENCIAS_MSMTIT AS INT64) AS TPN_TRANSFERENCIAS_MSMTIT60,
            CAST(M60.TPN_SAQUES AS INT64) AS TPN_SAQUES60,
            CAST(M90.TPN_SELLER AS INT64) AS TPN_SELLER90,
            CAST(M90.TPN_FUNDER AS INT64) AS TPN_FUNDER90,
            CAST(M90.TPN_RECIEVER AS INT64) AS TPN_RECIEVER90,
            CAST(M90.TPN_BORROWER AS INT64) AS TPN_BORROWER90,
            CAST(M90.TPN_COMPRAS_WEB AS INT64) AS TPN_COMPRAS_WEB90,
            CAST(M90.TPN_COMPRAS_APP AS INT64) AS TPN_COMPRAS_APP90,
            CAST(M90.TPN_PRODUTOS_BANCARIOS AS INT64) AS TPN_PRODUTOS_BANCARIOS90,
            CAST(M90.TPN_TRANSFERENCIAS_DIFFTIT AS INT64) AS TPN_TRANSFERENCIAS_DIFFTIT90,
            CAST(M90.TPN_TRANSFERENCIAS_MSMTIT AS INT64) AS TPN_TRANSFERENCIAS_MSMTIT90,
            CAST(M90.TPN_SAQUES AS INT64) AS TPN_SAQUES90,
            CAST(M30.TPV_SELLER AS FLOAT64) AS TPV_SELLER30,
            CAST(M30.TPV_FUNDER AS FLOAT64) AS TPV_FUNDER30,
            CAST(M30.TPV_RECIEVER AS FLOAT64) AS TPV_RECIEVER30,
            CAST(M30.TPV_BORROWER AS FLOAT64) AS TPV_BORROWER30,
            CAST(M30.TPV_COMPRAS_WEB AS FLOAT64) AS TPV_COMPRAS_WEB30,
            CAST(M30.TPV_COMPRAS_APP AS FLOAT64) AS TPV_COMPRAS_APP30,
            CAST(M30.TPV_PRODUTOS_BANCARIOS AS FLOAT64) AS TPV_PRODUTOS_BANCARIOS30,
            CAST(M30.TPV_TRANSFERENCIAS_DIFFTIT AS FLOAT64) AS TPV_TRANSFERENCIAS_DIFFTIT30,
            CAST(M30.TPV_TRANSFERENCIAS_MSMTIT AS FLOAT64) AS TPV_TRANSFERENCIAS_MSMTIT30,
            CAST(M30.TPV_SAQUES AS FLOAT64) AS TPV_SAQUES30,
            CAST(M60.TPV_SELLER AS FLOAT64) AS TPV_SELLER60,
            CAST(M60.TPV_FUNDER AS FLOAT64) AS TPV_FUNDER60,
            CAST(M60.TPV_RECIEVER AS FLOAT64) AS TPV_RECIEVER60,
            CAST(M60.TPV_BORROWER AS FLOAT64) AS TPV_BORROWER60,
            CAST(M60.TPV_COMPRAS_WEB AS FLOAT64) AS TPV_COMPRAS_WEB60,
            CAST(M60.TPV_COMPRAS_APP AS FLOAT64) AS TPV_COMPRAS_APP60,
            CAST(M60.TPV_PRODUTOS_BANCARIOS AS FLOAT64) AS TPV_PRODUTOS_BANCARIOS60,
            CAST(M60.TPV_TRANSFERENCIAS_DIFFTIT AS FLOAT64) AS TPV_TRANSFERENCIAS_DIFFTIT60,
            CAST(M60.TPV_TRANSFERENCIAS_MSMTIT AS FLOAT64) AS TPV_TRANSFERENCIAS_MSMTIT60,
            CAST(M60.TPV_SAQUES AS FLOAT64) AS TPV_SAQUES60,
            CAST(M90.TPV_SELLER AS FLOAT64) AS TPV_SELLER90,
            CAST(M90.TPV_FUNDER AS FLOAT64) AS TPV_FUNDER90,
            CAST(M90.TPV_RECIEVER AS FLOAT64) AS TPV_RECIEVER90,
            CAST(M90.TPV_BORROWER AS FLOAT64) AS TPV_BORROWER90,
            CAST(M90.TPV_COMPRAS_WEB AS FLOAT64) AS TPV_COMPRAS_WEB90,
            CAST(M90.TPV_COMPRAS_APP AS FLOAT64) AS TPV_COMPRAS_APP90,
            CAST(M90.TPV_PRODUTOS_BANCARIOS AS FLOAT64) AS TPV_PRODUTOS_BANCARIOS90,
            CAST(M90.TPV_TRANSFERENCIAS_DIFFTIT AS FLOAT64) AS TPV_TRANSFERENCIAS_DIFFTIT90,
            CAST(M90.TPV_TRANSFERENCIAS_MSMTIT AS FLOAT64) AS TPV_TRANSFERENCIAS_MSMTIT90,
            CAST(M90.TPV_SAQUES AS FLOAT64) AS TPV_SAQUES90,
            POPF.NR_SHOWNS_OPF AS SHOWNS_OPF,
            POPF.NR_OPENS_OPF AS OPENS_OPF,
            P30.NR_SHOWNS AS SHOWNS_30,
            P30.NR_OPENS AS OPENS_30,
            P60.NR_SHOWNS AS SHOWNS_60,
            P60.NR_OPENS AS OPENS_60,
            P90.NR_SHOWNS AS SHOWNS_90,
            P90.NR_OPENS AS OPENS_90
        FROM
            CUST_ATIVO AS CA
            LEFT JOIN CUSTS_OUT AS CO ON CA.cus_cust_id = CO.cus_cust_id
            LEFT JOIN PUSH_OPF AS POPF ON CA.cus_cust_id = POPF.cus_cust_id
            LEFT JOIN PUSH30 AS P30 ON CA.cus_cust_id = P30.cus_cust_id
            LEFT JOIN PUSH60 AS P60 ON CA.cus_cust_id = P60.cus_cust_id
            LEFT JOIN PUSH90 AS P90 ON CA.cus_cust_id = P90.cus_cust_id
            LEFT JOIN SALDO_CONTA30 AS SC30 ON CA.cus_cust_id = SC30.cus_cust_id
            LEFT JOIN SALDO_CONTA60 AS SC60 ON CA.cus_cust_id = SC60.cus_cust_id
            LEFT JOIN SALDO_CONTA90 AS SC90 ON CA.cus_cust_id = SC90.cus_cust_id
            LEFT JOIN CLUSTER_CUST AS CC ON CA.cus_cust_id = CC.cus_cust_id
            LEFT JOIN EMPLOYERS_MELI AS EM ON CA.cus_cust_id = EM.cus_cust_id
            LEFT JOIN PRINCIPALIDADE AS PR ON CA.cus_cust_id = PR.CUS_CUST_ID
            LEFT JOIN CUSTS_IN AS CI ON CA.cus_cust_id = CI.CUS_CUST_ID
            LEFT JOIN MAU30 AS M30 ON CA.cus_cust_id = M30.CUS_CUST_ID
            LEFT JOIN MAU60 AS M60 ON CA.cus_cust_id = M60.CUS_CUST_ID
            LEFT JOIN MAU90 AS M90 ON CA.cus_cust_id = M90.CUS_CUST_ID
            LEFT JOIN CADASTRO AS CAD ON CA.cus_cust_id = CAD.CUS_CUST_ID
            LEFT JOIN CARTAOCREDITO AS CART ON CA.cus_cust_id = CART.CUS_CUST_ID
            LEFT JOIN LOYALTY AS LYL ON CA.cus_cust_id = LYL.CUS_CUST_ID
            LEFT JOIN GEOLOC AS GEO ON CA.cus_cust_id = GEO.CUS_CUST_ID
            LEFT JOIN (
                SELECT * FROM ML_TBL.TB_PROPENSITY_SCORES
                WHERE safra=PARSE_DATE('%Y-%m-%d','{safra}')
                AND DT_EXEC=(SELECT max(DT_EXEC) FROM ML_TBL.TB_PROPENSITY_SCORES
                                WHERE SAFRA=PARSE_DATE('%Y-%m-%d','{safra}')
                                AND MODEL_NAME='{model_name}'
                            )
                AND MODEL_NAME='{model_name}'
                ) SCORES ON CA.CUS_CUST_ID = SCORES.CUS_CUST_ID
        WHERE
            STATUS = 'ACTIVE' --APP MP ATIVO
            -- AND SC30.AVAILABLE_BALANCE > 0 --POSSUI SALDO
            AND EM.cus_cust_id IS NULL --NÂO É FUNCIONARIO MELI
            AND P30.CUS_CUST_ID IS NOT NULL --RECEBEU PUSH 30 DIAS (NOT BLOKEDLIST)
            -- AND P30.NR_OPENS > 0 -- ABRIU ALGUM PUSH NOS ULTIMOS 30 DIAS
            AND CC.SEGMENT = 'SELLER'
            AND (
                CI.FLAG_IN_MES IS NULL
                OR CI.FLAG_IN_MES = 'S'
            )
            AND SCORES.CUS_CUST_ID IS NULL""".format(safra=safra_ini,model_name=model_name)

    df = bigquery.execute_response(query,  output="df")
    return df


