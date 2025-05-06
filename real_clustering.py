from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from Clustering import Conventional_Algorithm, RSC_Algorithm
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


def hdbscan_clustering(df):
    model = Conventional_Algorithm(df)
    df_with_labels = model.hdbscan_clustering()
    return df_with_labels


def rsc_clustering(df, k):
    rsc = RSC_Algorithm(k=k)
    labels = rsc.fit_predict(df.iloc[:, :2].values)
    df['labels'] = labels
    return df


def calculate_CVIs(df):
    SE = SEIndex(df)
    LCCV = LCCV_Index(df)
    DBCV = DBCV_Index(df)
    NCCV = NCCV_Index(df)
    CDbw = CDbwIndex(df)
    S_Dbw = S_Dbw_Index(df)
    XieBeni = XieBeniIndex(df)

    return {
            'DB': round(davies_bouldin_score(df.iloc[:, :-1], df['labels']), 3),
            'S_Dbw': round(S_Dbw.run(), 3),
            'Sil.': round(silhouette_score(df.iloc[:, :-1], df['labels']), 3),
            'Xie-Beni': round(XieBeni.run(), 3),
            'CDbw': round(CDbw.run(), 3),
            'DBCV': round(DBCV.run(), 3),
            'LCCV': round(LCCV.run(), 3),
            'NCCV': round(NCCV.run(), 3),
            'SE': round(SE.run(), 3)
            }


def read_all_datasets(read_datasets):
    dfs = []
    for name, info in read_datasets.items():
        data = read_data(info['category'], info['dataset'])
        dfs.append(data)
    return dfs
    

def evaluate_clustering_validity(cvi_dict):
    from collections import Counter

    preference = {
        'DB': 'min',         # Davies–Bouldin
        'S_Dbw': 'min',      # S_Dbw Index
        'Sil.': 'max',       # Silhouette Score
        'Xie-Beni': 'min',   # Xie–Beni Index
        'CDbw': 'min',       # Cluster Density Between-Within
        'DBCV': 'max',       # Density-Based Clustering Validation
        'LCCV': 'max',       # Local Clustering Coefficient Validity
        'NCCV': 'max',       # Normalized Clustering Coeff Validity
        'SE': 'max'          # Our proposed index
    }

    cvi_df = pd.DataFrame(cvi_dict).T

    best_per_metric = {}
    for metric, pref in preference.items():
        if pref == 'min':
            best_algo = cvi_df[metric].idxmin()
        else:
            best_algo = cvi_df[metric].idxmax()
        best_per_metric[metric] = best_algo

    # Count how many times each algorithm was the best
    best_counts = Counter(best_per_metric.values())
    total_best = pd.Series(best_counts).sort_values(ascending=False)

    return best_per_metric


def cvi_ari_calculator(read_datasets, dfs):
    ari_results, cvi_results, best_algorithm = {}, {}, {}
    dataset_values = list(read_datasets.values())
    dataset_names = list(read_datasets.keys())
    for i in range(len(dfs)):
        data = dfs[i]
        real_label = data['labels']
        df_KM = kmeans_clustering(data.iloc[:, :-1], dataset_values[i]['k'])
        df_Agg = agglomerative_clustering(data.iloc[:, :-1], dataset_values[i]['k'])
        df_OP = optics_clustering(data.iloc[:, :-1], dataset_values[i]['k']*3)
        df_HDB = hdbscan_clustering(data.iloc[:, :-1])
        df_RSC = rsc_clustering(data.iloc[:, :-1], dataset_values[i]['k'])

        KM_ARI = round(adjusted_rand_score(real_label, df_KM['labels']), 3)
        Agg_ARI = round(adjusted_rand_score(real_label, df_Agg['labels']), 3)
        OP_ARI = round(adjusted_rand_score(real_label, df_OP['labels']), 3)
        HDB_ARI = round(adjusted_rand_score(real_label, df_HDB['labels']), 3)
        RSC_ARI = round(adjusted_rand_score(real_label, df_RSC['labels']), 3)
        
        ari_results[dataset_names[i]] = {'KM-ARI':KM_ARI, 'Agg-ARI':Agg_ARI, 'OP-ARI':OP_ARI, 'HDB-ARI':HDB_ARI, 'RSC-ARI':RSC_ARI}

        if len(df_KM["labels"].unique()) > 1:
            data["labels"] = df_KM["labels"].values
            KM_CVI = calculate_CVIs(data)

        if len(df_Agg["labels"].unique()) > 1:
            data["labels"] = df_Agg["labels"].values
            Agg_CVI = calculate_CVIs(data)

        if len(df_OP["labels"].unique()) > 1:
            data["labels"] = df_OP["labels"].values
            OP_CVI = calculate_CVIs(data)

        if len(df_HDB["labels"].unique()) > 1:
            data["labels"] = df_HDB["labels"].values
            HDB_CVI = calculate_CVIs(data)

        if len(df_RSC["labels"].unique()) > 1:
            data["labels"] = df_RSC["labels"].values
            RSC_CVI = calculate_CVIs(data)
        
        cvi_results[dataset_names[i]] = {'KM-CVI':KM_CVI, 'Agg-CVI':Agg_CVI, 'OP-CVI':OP_CVI, 'HDB-CVI':HDB_CVI, 'RSC-CVI':RSC_CVI}
        best_algorithm[dataset_names[i]] = evaluate_clustering_validity({'KM-CVI':KM_CVI, 'Agg-CVI':Agg_CVI, 'OP-CVI':OP_CVI, 'HDB-CVI':HDB_CVI, 'RSC-CVI':RSC_CVI})
    
    ari_results = pd.DataFrame(ari_results).T
    ari_results['best_algorithm'] = ari_results.idxmax(axis=1)
    return ari_results, cvi_results, best_algorithm


uci_datasets = {
                 'Iris': {'category': 'uci', 'dataset': 'iris', 'k':3},
                 'Iris': {'category': 'uci', 'dataset': 'iris5', 'k':3},
                 'Wine': {'category': 'uci', 'dataset': 'wine', 'k':3},
                 'Glass': {'category': 'uci', 'dataset': 'glass', 'k':6},
                 'Breast Cancer': {'category': 'uci', 'dataset': 'breast-cancer', 'k':2},
                 'Seeds': {'category': 'uci', 'dataset': 'seeds', 'k':3},
                 'User Knowledge': {'category': 'uci', 'dataset': 'user-knowledge', 'k':2},
                 'Thyroid': {'category': 'uci', 'dataset': 'thyroid', 'k':6},
                 'Mammographic': {'category': 'uci', 'dataset': 'mammographic', 'k':2}
                }

sipu_datasets = {
                 'A1': {'category': 'sipu', 'dataset': 'a1', 'k':20},
                 'A2': {'category': 'sipu', 'dataset': 'a2', 'k':35},
                 'A3': {'category': 'sipu', 'dataset': 'a3', 'k':50},
                 'Aggregation': {'category': 'sipu', 'dataset': 'aggregation', 'k':7},
                 'Compound': {'category': 'sipu', 'dataset': 'compound', 'k':6},
                 'D31': {'category': 'sipu', 'dataset': 'd31', 'k':31},
                 'Flame': {'category': 'sipu', 'dataset': 'flame', 'k':2},
                 'Path-Based': {'category': 'sipu', 'dataset': 'pathbased', 'k':3},
                 'r15': {'category': 'sipu', 'dataset': 'r15', 'k':15},
                 'Jain': {'category': 'sipu', 'dataset': 'jain', 'k':2},
                 'S1': {'category': 'sipu', 'dataset': 's1', 'k':15},
                 'S2': {'category': 'sipu', 'dataset': 's2', 'k':15},
                 'S3': {'category': 'sipu', 'dataset': 's3', 'k':15},
                 'S4': {'category': 'sipu', 'dataset': 's4', 'k':15},
                 'Spiral': {'category': 'sipu', 'dataset': 'spiral', 'k':3},
                 'Unbalance': {'category': 'sipu', 'dataset': 'unbalance', 'k':8},
                 'Worms 2': {'category': 'sipu', 'dataset': 'worms_2', 'k':35},
                 'Worms 64': {'category': 'sipu', 'dataset': 'worms_64', 'k':25}
                }

fcps_datasets = {
                 'Atom': {'category': 'fcps', 'dataset': 'atom', 'k':2},
                 'Chain Link': {'category': 'fcps', 'dataset': 'chainlink', 'k':2},
                 'Engy time': {'category': 'fcps', 'dataset': 'engytime', 'k':2},
                 'Hepta': {'category': 'fcps', 'dataset': 'hepta', 'k':7},
                 'Lsun': {'category': 'fcps', 'dataset': 'lsun', 'k':3},
                 'Target': {'category': 'fcps', 'dataset': 'target', 'k':6},
                 'Tetra': {'category': 'fcps', 'dataset': 'tetra', 'k':4},
                 'Twodiamonds': {'category': 'fcps', 'dataset': 'twodiamonds', 'k':2},
                 'Wingnut': {'category': 'fcps', 'dataset': 'wingnut', 'k':2},
                }

graves_datasets = {
                 'Dense': {'category': 'graves', 'dataset': 'dense', 'k':2},
                 'Fuzzy x': {'category': 'graves', 'dataset': 'fuzzyx', 'k':5},
                 'line': {'category': 'graves', 'dataset': 'line', 'k':2},
                 'Parabolic': {'category': 'graves', 'dataset': 'parabolic', 'k':2},
                 'Ring Noisy': {'category': 'graves', 'dataset': 'ring_noisy', 'k':3},
                 'Ring Outliers': {'category': 'graves', 'dataset': 'ring_outliers', 'k':5},
                 'Ring': {'category': 'graves', 'dataset': 'ring', 'k':2},
                 'Zigzag Noisy': {'category': 'graves', 'dataset': 'zigzag_noisy', 'k':4},
                 'Zigzag Outliers': {'category': 'graves', 'dataset': 'zigzag_outliers', 'k':4},
                 'Zigzag': {'category': 'graves', 'dataset': 'zigzag', 'k':3}
                }

dfs = read_all_datasets(read_datasets)
ari_results, cvi_results, best_algorithm = cvi_ari_calculator(read_datasets, dfs)

print(pd.DataFrame(best_algorithm).T)
print()
print(ari_results)
