import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Load the Excel file
excel_file = 'Datasets/Synthetic Datasets.xlsx'

# Create a figure with a 5x5 grid of subplots
fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# Loop through the sheets and plot each dataset
for i, ax in enumerate(axes.flatten(), start=1):
    # Read the sheet into a DataFrame
    df = pd.read_excel(excel_file, sheet_name=f'df{i}')
    
    # Scatter plot for each dataset with alpha=0.8 for transparency
    sns.scatterplot(data=df, x='x', y='y', color='#C9234B', ax=ax, legend=None, s=25, edgecolor='#C9234B', alpha=0.7)
    
    # Set the title for each subplot
    ax.set_title(f'Dataset {i}')
    
    # Remove "x" and "y" labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Set aspect ratio to 1:1 for each subplot
    ax.set_aspect('equal')

# Adjust layout to avoid overlapping labels and ensure equal sizing for all plots
plt.tight_layout(pad=3)  # Increase general padding between plots

# Decrease the horizontal padding specifically
plt.subplots_adjust(hspace=0.5, wspace=0)  # Custom horizontal and vertical spacing

# Show the plot
plt.show()
