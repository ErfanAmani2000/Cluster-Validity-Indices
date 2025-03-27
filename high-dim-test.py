# from sklearn.datasets import fetch_openml
# import numpy as np

# mnist = fetch_openml('mnist_784', version=1)
# X = mnist.data.to_numpy()  # 70000 samples, 784 features
# y = mnist.target.to_numpy()

# --------------------------------------------------------

import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
df = pd.read_csv(url, compression='gzip', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values