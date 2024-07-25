from sklearn.metrics import davies_bouldin_score, silhouette_score
from Clustering import RSC_Algorithm, Conventional_Algorithm
from sklearn.preprocessing import LabelEncoder
from S_Dbw import S_Dbw_Index
from CDbw import CDbw_Index
from DBCV import DBCV_Index
from LCCV import LCCV_Index
from SE import SE_Index
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
    SE = SE_Index(df)
    LCCV = LCCV_Index(df)
    DBCV = DBCV_Index(df)
    CDbw = CDbw_Index(df)
    S_Dbw = S_Dbw_Index(df)

    return {
            'DB': davies_bouldin_score(df.iloc[:, :-1], df.iloc[:, -1]),
            'S_Dbw': S_Dbw.run(),
            'Sil.': silhouette_score(df.iloc[:, :-1], df.iloc[:, -1]),
            'CDbw': CDbw.run(),
            'DBCV': DBCV.run(),
            'LCCV': LCCV.run(),
            'SE': SE.run()
            }


start_time = time.time()

url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
df = data_get(url)
sample_df = df.sample(n=10000, random_state=1)
SE = SE_Index(sample_df)
print(f"CVI Measure: {SE.run()}")

end_time = time.time()
print(f"CPU Time: {end_time-start_time:.3f}")
