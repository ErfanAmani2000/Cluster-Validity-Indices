from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from CVIs.XieBeni import XieBeniIndex
from CVIs.S_Dbw import S_Dbw_Index
from CVIs.CDbw import CDbwIndex
from CVIs.DBCV import DBCV_Index
from CVIs.LCCV import LCCV_Index
from CVIs.NCCV import NCCV_Index
from CVIs.SE import SEIndex
import pandas as pd
import numpy as np
import gzip


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

    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    df = df.copy()
    for col in df.columns:
        nan_percentage = (df[col].isna().sum() / len(df)) * 100
        if nan_percentage > 10:
            df.drop(columns=[col], inplace=True)
        else:
            if df[col].dtype == 'O':
                df.loc[:, col] = df[col].fillna(df[col].mode()[0])
            else: 
                df.loc[:, col] = df[col].fillna(df[col].mean())

    features = df.drop(columns=['labels'])
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    standardized_df = pd.DataFrame(standardized_features, columns=features.columns)
    standardized_df['labels'] = df['labels'].values
    return df


def calculate_CVIs_with_ARI(df):
    # Calculate the clustering validity indices (CVI)
    SE = SEIndex(df)
    LCCV = LCCV_Index(df)
    DBCV = DBCV_Index(df)
    NCCV = NCCV_Index(df)
    CDbw = CDbwIndex(df)
    S_Dbw = S_Dbw_Index(df)
    XieBeni = XieBeniIndex(df)
    
    # Get the predicted clusters for each CVI (assuming they have a 'run' method)
    SE_clusters = SE.run()  # Get predicted cluster labels (you'll need to modify this based on CVI's behavior)
    LCCV_clusters = LCCV.run()
    DBCV_clusters = DBCV.run() 
    NCCV_clusters = NCCV.run()
    CDbw_clusters = CDbw.run() 
    S_Dbw_clusters = S_Dbw.run()
    XieBeni_clusters = XieBeni.run()  
    
    # Calculate ARI for each CVI
    ARI_scores = {
        'DB': round(adjusted_rand_score(df['labels'], df['labels']), 3),  # ARI for true labels (should always be 1)
        'S_Dbw': round(adjusted_rand_score(df['labels'], S_Dbw_clusters), 3),
        'Sil.': round(adjusted_rand_score(df['labels'], df['labels']), 3),  # ARI should not be calculated for Silhouette
        'Xie-Beni': round(adjusted_rand_score(df['labels'], XieBeni_clusters), 3),
        'CDbw': round(adjusted_rand_score(df['labels'], CDbw_clusters), 3),
        'DBCV': round(adjusted_rand_score(df['labels'], DBCV_clusters), 3),
        'LCCV': round(adjusted_rand_score(df['labels'], LCCV_clusters), 3),
        'NCCV': round(adjusted_rand_score(df['labels'], NCCV_clusters), 3),
        'SE': round(adjusted_rand_score(df['labels'], SE_clusters), 3)
    }
    return ARI_scores

df = read_data(category='uci', dataset='iris')
ARI_scores = calculate_CVIs_with_ARI(df)

print(ARI_scores)
