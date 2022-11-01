from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


def main():
    dataset = pd.read_csv('dataset.csv', index_col='id')
    x = np.array(dataset.drop(['Class'], axis=1))
    y = np.array(dataset.Class)
    k = 3
    neigh = KNeighborsClassifier(n_neighbors=k, p=2)
    neigh.fit(x, y)
    new = np.array([74, 92]).reshape(1, -1)
    print("Euclidean distance")
    print("Distance  from", new, "to nearest neighbor: ", neigh.kneighbors(n_neighbors=1, X=new))
    print(k, "nearest neighbors for", new, ":", neigh.kneighbors(n_neighbors=k, X=new, return_distance=False))
    print("Predicted class for", new, "with k =", k, ":", neigh.predict(new))
    print('')
    print("Taxicab geometry")
    neigh2 = KNeighborsClassifier(n_neighbors=k, p=1)
    neigh2.fit(x, y)
    print("Distance  from", new, "to nearest neighbor: ", neigh2.kneighbors(n_neighbors=1, X=new))
    print(k, "nearest neighbors for", new, ":", neigh2.kneighbors(n_neighbors=k, X=new, return_distance=False))
    print("Predicted class for", new, "with k =", k, ":", neigh2.predict(new))


if __name__ == '__main__':
    main()

