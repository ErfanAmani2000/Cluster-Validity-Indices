import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

# Define the success counts and total number of datasets for each CVI
data = {
    'CVI': ['DB', 'S_Dbw', 'Sil.', 'Xie-Beni', 'CDbw', 'DBCV', 'LCCV', 'NCCV', 'SE'],
    'Success_Count': [29, 13, 27, 34, 14, 38, 41, 50, 58],
    'Total_Datasets': [71, 71, 71, 71, 71, 71, 71, 71, 71],
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Calculate success rates (percentages)
df['Success_Rate'] = df['Success_Count'] / df['Total_Datasets'] * 100

# Display success rates
print("Success Rates for Each CVI:")
print(df[['CVI', 'Success_Rate']])

# Pairwise Proportions Test between SE and other CVIs
for index, row in df.iterrows():
    if row['CVI'] != 'SE':
        # Compare SE to each other CVI
        se_success_count = df[df['CVI'] == 'SE']['Success_Count'].values[0]
        se_total = df[df['CVI'] == 'SE']['Total_Datasets'].values[0]
        
        # Success count for the other CVI
        other_success_count = row['Success_Count']
        other_total = row['Total_Datasets']
        
        # Perform proportions Z-test
        successes = np.array([se_success_count, other_success_count])
        totals = np.array([se_total, other_total])
        
        stat, p_value = proportions_ztest(successes, totals)
        
        print(f"Proportions Z-Test for SE vs {row['CVI']}: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"Significant difference found between SE and {row['CVI']}")
        else:
            print(f"No significant difference found between SE and {row['CVI']}")

# Chi-Squared Test for overall distribution of successes across CVIs
success_counts = df['Success_Count'].values
total_counts = df['Total_Datasets'].values

# Create contingency table (success vs failure for each CVI)
successes = np.array(success_counts)
failures = np.array(total_counts - success_counts)
contingency_table = np.array([successes, failures])

# Perform Chi-Squared Test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table.T)  # Transpose for proper shape

print(f"\nChi-Squared Test: chi2_stat = {chi2_stat:.4f}, p-value = {p_value:.4f}, dof = {dof}")
