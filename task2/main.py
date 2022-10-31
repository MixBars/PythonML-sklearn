from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import seaborn as sns


def main():
    #импорт данных
    dataset = pd.read_csv('37_25.csv', header=None)
    #центрирование
    dataset = dataset - dataset.mean(axis=0)

    #инициализация МГК
    pca_test = PCA(n_components=2, svd_solver='full')
    pca_test.fit(dataset)
    dataset = pca_test.transform(dataset)
    print(dataset)

    #визуализация МГК
    for i in range(60):
        plt.scatter(dataset[i][0], dataset[i][1])
    plt.show()

    #подсчет объясненной дисперсии
    explained_variance = pca_test.explained_variance_ratio_
    sum = 0
    for i in range(len(explained_variance)):
        sum += explained_variance[i]
    print(sum)

    #восстановление логотипа. Умножаем матрицу счета на матрицу весов транспонироваанную
    z = pd.read_csv('X_reduced_456.csv', header=None, sep=';')
    print(z)
    fi = pd.read_csv('X_loadings_456.csv', header=None, sep=';')
    fi = np.array(fi).transpose()

#отрисовка логотипа
#    result = pd.DataFrame(np.matmul(z, fi))
#    for i in range(100):
#        for j in range(100):
#            if result[i][j] > 0:
#                plt.scatter(i,j)
#    plt.show()

if __name__ == '__main__':
    main()
