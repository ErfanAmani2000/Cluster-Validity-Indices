import pandas as pd
from sklearn.metrics import adjusted_rand_score
from Clustering import Conventional_Algorithm, RSC_Algorithm

sheets = {f'df{i}': pd.read_excel('Datasets/Synthetic Datasets.xlsx', sheet_name=f'df{i}') for i in range(1, 26)}

df1 = sheets['df1'].iloc[:, 1:]
df2 = sheets['df2'].iloc[:, 1:]
df3 = sheets['df3'].iloc[:, 1:]
df4 = sheets['df4'].iloc[:, 1:]
df5 = sheets['df5'].iloc[:, 1:]
df6 = sheets['df6'].iloc[:, 1:]
df7 = sheets['df7'].iloc[:, 1:]
df8 = sheets['df8'].iloc[:, 1:]
df9 = sheets['df9'].iloc[:, 1:]
df10 = sheets['df10'].iloc[:, 1:]
df11 = sheets['df11'].iloc[:, 1:]
df12 = sheets['df12'].iloc[:, 1:]
df13 = sheets['df13'].iloc[:, 1:]
df14 = sheets['df14'].iloc[:, 1:]
df15 = sheets['df15'].iloc[:, 1:]
df16 = sheets['df16'].iloc[:, 1:]
df17 = sheets['df17'].iloc[:, 1:]
df18 = sheets['df18'].iloc[:, 1:]
df19 = sheets['df19'].iloc[:, 1:]
df20 = sheets['df20'].iloc[:, 1:]
df21 = sheets['df21'].iloc[:, 1:]
df22 = sheets['df22'].iloc[:, 1:]
df23 = sheets['df23'].iloc[:, 1:]
df24 = sheets['df24'].iloc[:, 1:]
df25 = sheets['df25'].iloc[:, 1:]

datasets = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14,
              df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25]

results = []

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

model_table = [
    ['KM(2)', 'Agg(5)', 'KM(4)', 'OP(5)', 'KM(3)', 'RSC(3)'],
    ['KM(2)', 'Agg(5)', 'KM(4)', 'OP(5)', 'KM(3)', 'RSC(3)'],
    ['KM(2)', 'Agg(5)', 'KM(4)', 'OP(5)', 'KM(3)', 'RSC(3)'],
    ['KM(2)', 'Agg(2)', 'HDB(5)', 'OP(14)', 'KM(3)', 'RSC(2)'],
    ['KM(2)', 'Agg(3)', 'HDB(3)', 'HDB(6)', 'KM(3)', 'RSC(3)'],
    ['KM(2)', 'Agg(2)', 'HDB(6)', 'HDB(3)', 'KM(3)', 'RSC(3)'],
    ['HDB(31)', 'Agg(4)', 'HDB(10)', 'HDB(5)', 'HDB(18)', 'RSC(4)'],
    ['KM(2)', 'Agg(3)', 'KM(3)', 'KM(6)', 'KM(4)', 'RSC(3)'],
    ['KM(2)', 'Agg(3)', 'HDB(5)', 'KM(6)', 'HDB(48)', 'RSC(4)'],
    ['KM(2)', 'Agg(3)', 'HDB(5)', 'KM(6)', 'HDB(21)', 'RSC(3)'],
    ['OP(22)', 'Agg(3)', 'KM(10)', 'HDB(5)', 'KM(5)', 'RSC(4)'],
    ['OP(22)', 'Agg(3)', 'KM(10)', 'HDB(5)', 'KM(9)', 'RSC(9)'],
    ['HDB(5)', 'Agg(3)', 'KM(3)', 'HDB(40)', 'KM(6)', 'RSC(3)'],
    ['HDB(5)', 'Agg(3)', 'KM(3)', 'KM(4)', 'KM(6)', 'RSC(4)'],
    ['HDB(5)', 'Agg(6)', 'KM(3)', 'KM(6)', 'KM(12)', 'RSC(6)'],
    ['KM(3)', 'Agg(3)', 'KM(2)', 'KM(6)', 'HDB(5)', 'RSC(3)'],
    ['KM(3)', 'Agg(3)', 'KM(2)', 'KM(6)', 'OP(4)', 'RSC(2)'],
    ['KM(2)', 'Agg(3)', 'KM(4)', 'KM(3)', 'HDB(8)', 'RSC(2)'],
    ['KM(4)', 'Agg(3)', 'KM(2)', 'KM(15)', 'OP(10)', 'RSC(4)'],
    ['KM(4)', 'Agg(6)', 'KM(2)', 'KM(15)', 'KM(12)', 'RSC(6)'],
    ['KM(4)', 'Agg(6)', 'KM(2)', 'KM(15)', 'HDB(5)', 'RSC(4)'],
    ['KM(4)', 'Agg(6)', 'KM(2)', 'KM(15)', 'HDB(5)', 'RSC(3)'],
    ['KM(4)', 'Agg(6)', 'KM(2)', 'KM(15)', 'HDB(25)', 'RSC(3)'],
    ['KM(4)', 'Agg(3)', 'KM(2)', 'KM(15)', 'HDB(5)', 'RSC(3)'],
    ['KM(4)', 'Agg(3)', 'KM(2)', 'KM(15)', 'HDB(20)', 'RSC(2)']
]

results = []

# Iterate over datasets and model configurations
for i, df in enumerate(datasets):
    dataset_id = i + 1  # To match dataset 1, 2, 3, ..., 25
    row = {'dataset': dataset_id}
    
    # For each model configuration, apply the corresponding clustering
    for j, model_config in enumerate(model_table[i]):
        model_name, param = model_config.split('(')
        param = int(param[:-1])  # Extract the parameter number
        
        # Apply the appropriate clustering function based on model name
        if model_name == 'KM':
            df_with_labels = kmeans_clustering(df.copy(), param)
        elif model_name == 'Agg':
            df_with_labels = agglomerative_clustering(df.copy(), param)
        elif model_name == 'HDB':
            df_with_labels = hdbscan_clustering(df.copy())
        elif model_name == 'OP':
            df_with_labels = optics_clustering(df.copy(), param)
        elif model_name == 'RSC':
            df_with_labels = rsc_clustering(df.copy(), param)
        
        # Calculate ARI
        ari = adjusted_rand_score(df['real labels'], df_with_labels['labels'])
        
        # Store the ARI result for the model in the corresponding "Model" column
        row[f'Model {j + 1}'] = ari
    
    # Append the row (for the current dataset) to the results list
    results.append(row)

# Create a DataFrame to store the results
result_df = pd.DataFrame(results)

# Set the dataset column as index
result_df = result_df.set_index('dataset')

print(result_df)
