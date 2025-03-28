from sklearn.metrics import davies_bouldin_score, silhouette_score
from Clustering import RSC_Algorithm, Conventional_Algorithm
from sklearn.preprocessing import LabelEncoder
from S_Dbw import S_Dbw_Index
from CDbw import CDbwIndex
from DBCV import DBCV_Index
from LCCV import LCCV_Index
from NCCV import NCCV_Index
from SE import SEIndex
import pandas as pd
import numpy as np
import time


def data_get(url):
    df = pd.read_csv(url, header=None, compression='gzip')
    categorical_columns = df.select_dtypes(include=['object']).columns

    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    df.rename(columns={df.columns[-1]: 'labels'}, inplace=True)
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

start_time = time.time()

url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
df = data_get(url)
sample_df = df.sample(n=100, random_state=1)
# SE = SEIndex(sample_df)
# print(f"CVI Measure: {SE.run()}")

# end_time = time.time()
# print(f"CPU Time: {end_time-start_time:.3f}")

print(calculate_CVIs(sample_df))
