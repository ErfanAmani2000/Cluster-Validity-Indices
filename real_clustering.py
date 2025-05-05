from Clustering import Conventional_Algorithm, RSC_Algorithm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score
from CVIs.XieBeni import XieBeniIndex
from CVIs.S_Dbw import S_Dbw_Index
from CVIs.CDbw import CDbwIndex
from CVIs.DBCV import DBCV_Index
from CVIs.LCCV import LCCV_Index
from CVIs.NCCV import NCCV_Index
from CVIs.SE import SEIndex
import pandas as pd
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
    return standardized_df

def kmeans_clustering(df, n_clusters):
    model = Conventional_Algorithm(df)
    df_with_labels = model.kmeans_clustering(n_clusters)
    return df_with_labels

def agglomerative_clustering(df, n_clusters):
    model = Conventional_Algorithm(df)
    df_with_labels = model.agglomerative_clustering(n_clusters)
    return df_with_labels

def optics_clustering(df, min_samples):
    model = Conventional_Algorithm(df)
    df_with_labels = model.optics_clustering(min_samples)
    return df_with_labels

def dbscan_clustering(df, min_samples):
    model = Conventional_Algorithm(df)
    df_with_labels = model.dbscan_clustering(min_samples)
    return df_with_labels

def rsc_clustering(df, k):
    rsc = RSC_Algorithm(k=k)
    labels = rsc.fit_predict(df.iloc[:, :2].values)
    df['labels'] = labels
    return df

# def calculate_CVIs(df):
#     SE = SEIndex(df)
#     LCCV = LCCV_Index(df)
#     DBCV = DBCV_Index(df)
#     NCCV = NCCV_Index(df)
#     CDbw = CDbwIndex(df)
#     S_Dbw = S_Dbw_Index(df)
#     XieBeni = XieBeniIndex(df)

#     return {
#             'DB': round(davies_bouldin_score(df.iloc[:, :-1], df['labels']), 3),
#             'S_Dbw': round(S_Dbw.run(), 3),
#             'Sil.': round(silhouette_score(df.iloc[:, :-1], df['labels']), 3),
#             'Xie-Beni': round(XieBeni.run(), 3),
#             'CDbw': round(CDbw.run(), 3),
#             'DBCV': round(DBCV.run(), 3),
#             'LCCV': round(LCCV.run(), 3),
#             'NCCV': round(NCCV.run(), 3),
#             'SE': round(SE.run(), 3)
#             }


read_datasets = {'Iris': {'category': 'uci', 'dataset': 'iris', 'k':3},
                 'Wine': {'category': 'uci', 'dataset': 'wine', 'k':3},
                 'Glass': {'category': 'uci', 'dataset': 'glass', 'k':6},
                 'Aggregation': {'category': 'sipu', 'dataset': 'aggregation', 'k':7},
                 'Jain': {'category': 'sipu', 'dataset': 'jain', 'k':2},
                 'Spiral': {'category': 'sipu', 'dataset': 'spiral', 'k':3},
                 'Unbalance': {'category': 'sipu', 'dataset': 'unbalance', 'k':8},
                 'Atom': {'category': 'fcps', 'dataset': 'atom', 'k':2},
                 'Chain Link': {'category': 'fcps', 'dataset': 'chainlink', 'k':2},
                 'Hepta': {'category': 'fcps', 'dataset': 'hepta', 'k':7},
                 'Lsun': {'category': 'fcps', 'dataset': 'lsun', 'k':3},
                 'Tetra': {'category': 'fcps', 'dataset': 'tetra', 'k':4},
                 'Twodiamonds': {'category': 'fcps', 'dataset': 'twodiamonds', 'k':2},
                 'Wingnut': {'category': 'fcps', 'dataset': 'wingnut', 'k':2},
                 'Dense': {'category': 'graves', 'dataset': 'dense', 'k':2},
                 'line': {'category': 'graves', 'dataset': 'line', 'k':2},
                 'Ring Noisy': {'category': 'graves', 'dataset': 'ring_noisy', 'k':3},
                 'Ring': {'category': 'graves', 'dataset': 'ring', 'k':2}}

dfs = []
for name, info in read_datasets.items():
    df = read_data(info['category'], info['dataset'])
    dfs.append(df)

dataset_values = list(read_datasets.values())
dataset_names = list(read_datasets.keys())
for i in range(len(dfs)):
    df = dfs[i]
    df_KM = kmeans_clustering(df.iloc[:, :-1], dataset_values[i]['k'])
    df_Agg = agglomerative_clustering(df.iloc[:, :-1], dataset_values[i]['k'])
    df_OP = optics_clustering(df.iloc[:, :-1], dataset_values[i]['k']*4)
    df_HDB = dbscan_clustering(df.iloc[:, :-1], dataset_values[i]['k']*3)
    df_RSC = rsc_clustering(df.iloc[:, :-1], dataset_values[i]['k'])

    df['KM-labels'] = df_KM['labels']
    df['Agg-labels'] = df_Agg['labels']
    df['OP-labels'] = df_OP['labels']
    df['HDB-labels'] = df_HDB['labels']
    df['RSC-labels'] = df_RSC['labels']
    dfs[i] = df

results = {}
for i in range(len(dfs)):
    df = dfs[i]
    real_label = df['labels']
    KM_ARI = round(adjusted_rand_score(real_label, df['KM-labels']), 3)
    Agg_ARI = round(adjusted_rand_score(real_label, df['Agg-labels']), 3)
    OP_ARI = round(adjusted_rand_score(real_label, df['OP-labels']), 3)
    HDB_ARI = round(adjusted_rand_score(real_label, df['HDB-labels']), 3)
    RSC_ARI = round(adjusted_rand_score(real_label, df['RSC-labels']), 3)
    results[dataset_names[i]] = {'KM-ARI':KM_ARI, 'Agg-ARI':Agg_ARI, 'OP-ARI':OP_ARI, 'HDB-ARI':HDB_ARI, 'RSC-ARI':RSC_ARI}
    
print(pd.DataFrame(results).T)
