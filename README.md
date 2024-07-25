# Cluster Validity Indices and Clustering Algorithms

This repository contains implementations of various clustering algorithms and cluster validity indices, including a novel SE measure. The main purpose of this project is to provide an efficient tool for clustering analysis and validation, with a particular focus on the new SE cluster validity measure.

## Abstract

Due to the absence of labels or so-called target variables, clustering validation, unlike classification, is more complex. Thus, cluster evaluation is challenging both in research projects and applications. While many clustering validity indices are addressed in the literature, most of them, even those widely used in applications, cannot handle arbitrary shapes.

This paper proposes a novel clustering validity index that can accommodate arbitrary shapes and is much more powerful in capturing the data's structure. The index is based on the significance of separation, for which a novel separation measure is proposed. In this approach, clusters that are significantly separated from each other take a maximum value of one. This threshold ensures that sparseness does not lose its importance and contribution to calculating the score for the cluster validity index.

Extensive experiments are conducted on 33 natural and synthetic datasets, revealing the promising efficacy of the proposed algorithm. Contrary to some existing density-based clustering validity indices, the computational expensiveness test, again performed on the 33 datasets mentioned above, demonstrates the proposed index's applicability even on large-scale datasets.

## Features

- **Clustering Algorithms**:
  - K-Means Clustering
  - Agglomerative Clustering
  - OPTICS Clustering
  - DBSCAN Clustering
  - RSC Algorithm (Novel Clustering Algorithm)

- **Cluster Validity Indices**:
  - Davies-Bouldin Score
  - Silhouette Score
  - S_Dbw Index
  - CDbw Index
  - LCCV Index
  - DBCV Index
  - SE Measure (Novel Cluster Validity Measure)

## Installation

To use the code in this repository, clone it to your local machine and install the necessary dependencies.

```sh
git clone https://github.com/your-username/cluster-validity-indices.git
cd cluster-validity-indices
