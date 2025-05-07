from sklearn.metrics import davies_bouldin_score, silhouette_score
from Clustering import RSC_Algorithm, Conventional_Algorithm
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from CVIs.XieBeni import XieBeniIndex
from CVIs.S_Dbw import S_Dbw_Index
from CVIs.CDbw import CDbwIndex
from CVIs.DBCV import DBCV_Index
from CVIs.LCCV import LCCV_Index
from CVIs.NCCV import NCCV_Index
from CVIs.SE import SEIndex
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
    XieBeni = XieBeniIndex(df)

    return {
            'DB': davies_bouldin_score(df.iloc[:, :-1], df.iloc[:, -1]),
            'S_Dbw': S_Dbw.run(),
            'Sil': silhouette_score(df.iloc[:, :-1], df.iloc[:, -1]),
            'Xie_Beni': XieBeni.run(),  # Changed column name to avoid '-'
            'CDbw': CDbw.run(),
            'DBCV': DBCV.run(),
            'LCCV': LCCV.run(),
            'NCCV': NCCV.run(),
            'SE': SE.run()
            }


# Generate synthetic data
def generate_synthetic_data(n_samples=100, n_features=2, n_clusters=3, noise_level=0.1):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    df = pd.DataFrame(X_noisy, columns=[f'feature_{i}' for i in range(n_features)])
    df['labels'] = y
    return df


# Introduce outliers
def add_outliers(df, outlier_ratio=0.1):
    n_outliers = int(outlier_ratio * len(df))
    outliers = np.random.uniform(low=df.min().min(), high=df.max().max(), size=(n_outliers, df.shape[1]-1))
    outliers_df = pd.DataFrame(outliers, columns=df.columns[:-1])
    outliers_df['labels'] = -1  # Label for outliers
    df_with_outliers = pd.concat([df, outliers_df], ignore_index=True)
    return df_with_outliers


# Conduct experiments
def conduct_experiments(noise_levels=[0.1, 0.3, 0.5], outlier_ratios=[0.1, 0.2]):
    results = []

    for noise_level in noise_levels:
        for outlier_ratio in outlier_ratios:
            # Generate noisy data
            df_noisy = generate_synthetic_data(noise_level=noise_level)
            # Add outliers
            df_noisy_with_outliers = add_outliers(df_noisy, outlier_ratio=outlier_ratio)
            # Calculate CVIs
            cvi_results = calculate_CVIs(df_noisy_with_outliers)
            results.append({
                'noise_level': noise_level,
                'outlier_ratio': outlier_ratio,
                **cvi_results
            })
    
    return pd.DataFrame(results)


# Run the experiments
experiment_results = conduct_experiments()


# One-way ANOVA for each CVI with respect to noise_level
anova_results = {}
for cvi in experiment_results.columns[2:]:
    f_val, p_val = f_oneway(experiment_results[experiment_results['noise_level'] == 0.1][cvi],
                            experiment_results[experiment_results['noise_level'] == 0.3][cvi],
                            experiment_results[experiment_results['noise_level'] == 0.5][cvi])
    anova_results[cvi] = (f_val, p_val)

# Display ANOVA results
print("ANOVA Results (F-value, p-value):")
for cvi, (f_val, p_val) in anova_results.items():
    print(f"{cvi}: {f_val}, {p_val}")


# Convert to DataFrame
df = pd.DataFrame(experiment_results)


# Normalize each CVI column using min-max normalization
normalized_df = df.copy()
for cvi in df.columns[2:]:
    min_val = df[cvi].min()
    max_val = df[cvi].max()
    normalized_df[cvi] = (df[cvi] - min_val) / (max_val - min_val)


# One-way ANOVA for each normalized CVI with respect to noise_level
anova_results = {}
for cvi in normalized_df.columns[2:]:
    model = ols(f'{cvi} ~ C(noise_level)', data=normalized_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_results[cvi] = anova_table


# Display ANOVA results
print("ANOVA Results:")
for cvi, anova_table in anova_results.items():
    print(f"\n{cvi} ANOVA Table:")
    print(anova_table)


# Tukey's HSD test for each normalized CVI
print("\nTukey's HSD Test Results:")
for cvi in normalized_df.columns[2:]:
    tukey = pairwise_tukeyhsd(endog=normalized_df[cvi],
                              groups=normalized_df['noise_level'],
                              alpha=0.05)
    print(f"\nTukey's HSD Results for {cvi}:")
    print(tukey)


# Melt the DataFrame for easier plotting
melted_df = normalized_df.melt(id_vars=['noise_level', 'outlier_ratio'], 
                               value_vars=df.columns[2:], 
                               var_name='CVI', 
                               value_name='Value')

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create box plots
sns.boxplot(x='CVI', y='Value', data=melted_df)

# Add title and labels
plt.title('Comparison of Normalized CVI Measures')
plt.xlabel('CVI Measure')
plt.ylabel('Normalized Value')

# Show plot
plt.tight_layout()
plt.show()
