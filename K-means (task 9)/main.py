from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def main():
    dataset = pd.read_csv("dataset.csv", index_col="Object").drop(["Cluster"], axis=1)
    kmeans = KMeans(n_clusters=3, init=np.array([[10.33, 8.5], [10.0, 7.0], [12.57, 12.14]]), max_iter=100, n_init=1)
    kmeans.fit(dataset)
    print("Clusters for objects: ", kmeans.labels_)
    print("Cluster centers:\n", kmeans.cluster_centers_)


if __name__ == '__main__':
    main()

