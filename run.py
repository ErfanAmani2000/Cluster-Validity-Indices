from sklearn.metrics import davies_bouldin_score, silhouette_score
from Clustering import RSC_Algorithm, Conventional_Algorithm
from S_Dbw import S_Dbw_Index
from CDbw import CDbw_Index
from DBCV import DBCV_Index
from LCCV import LCCV_Index
from SE import SE_Index
import pandas as pd
import numpy as np


data = np.concatenate((np.random.randn(100, 2) + [5, 5], 
                       np.random.randn(100, 2) + [2, 2], 
                       np.random.randn(100, 2) + [2, 5], 
                       np.random.randn(100, 2) + [4, 8]))


Algorithms = Conventional_Algorithm(pd.DataFrame(data))
df = Algorithms.kmeans_clustering(4)
df.columns = ['X', 'Y', 'labels']


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

print(calculate_CVIs(df))
