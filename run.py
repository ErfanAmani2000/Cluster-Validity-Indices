from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from CVIs.S_Dbw import S_Dbw_Index
from CVIs.CDbw import CDbwIndex
from CVIs.DBCV import DBCV_Index
from CVIs.LCCV import LCCV_Index
from CVIs.NCCV import NCCV_Index
from CVIs.SE import SEIndex
import pandas as pd
import numpy as np
import gzip
import time


def read_data(category, dataset):
    def read_gz_file(file_path, delimiter=' '):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f, delimiter=delimiter, header=None)
        return df

    df = read_gz_file(f"Datasets/{category}/{dataset}.data.gz")
    labels_df = read_gz_file(f"Datasets/{category}/{dataset}.labels0.gz")
    feature_names = [f'feature_{i}' for i in range(1, df.shape[1] + 1)]
    df.columns = feature_names

    labels_df = labels_df.iloc[:, -1]
    df['labels'] = labels_df.values
    return df


def calculate_CVIs(df):
    SE = SEIndex(df)
    LCCV = LCCV_Index(df)
    DBCV = DBCV_Index(df)
    NCCV = NCCV_Index(df)
    CDbw = CDbwIndex(df)
    S_Dbw = S_Dbw_Index(df)

    return {
            'DB': round(davies_bouldin_score(df.iloc[:, :-1], df.iloc[:, -1]), 3),
            'S_Dbw': round(S_Dbw.run(), 3),
            'Sil.': round(silhouette_score(df.iloc[:, :-1], df.iloc[:, -1]), 3),
            'CDbw': round(CDbw.run(), 3),
            'DBCV': round(DBCV.run(), 3),
            'LCCV': round(LCCV.run(), 3),
            'NCCV': round(NCCV.run(), 3),
            'SE': round(SE.run(), 3)
            }

df = read_data(category='uci', dataset='wine')

# start_time = time.time()
print(calculate_CVIs(df))
# end_time = time.time()
# print(f'CPU time: {end_time - start_time:2f}')
