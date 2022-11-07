import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


def main():
    #Скачиваем набор данных
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #Проверяем, что в датасетах все верно
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000, )
    #Решейп трехмерного массива в двумерный
    dataset = x_train.reshape(60000, 784)
    #Инициализируем МГК для всего возможного кол-ва ГК
    pca = PCA(n_components=784, svd_solver='full')
    pca.fit(dataset)
    #ВАРИАНТ 1: Найдем в цикле кол-во ГК для доли объясненной дисперсии 0.83
    explained_variance = pca.explained_variance_ratio_
    counter = 0
    i = 0
    while counter < 0.83:
        counter += explained_variance[i]
        i += 1
    M = i
    print("M =", i)
    #ВАРИАНТ 2: Рисуем график объясненной дисперсии + линию отсечения на границе 0.83
    for i in range(1, len(explained_variance)):
        explained_variance[i] += explained_variance[i - 1]
    plt.plot(np.arange(1, 785), explained_variance)
    plt.plot(np.zeros(784)+0.83, color='red')
    plt.show()
    plt.clf()
    M = int(input("Enter M: "))
    #Найдя кол-во ГК для доли объясненной дисперсии > 0.83, переинициализируем МГК
    pca = PCA(n_components=M, svd_solver='full')
    pca.fit(dataset)
    #Новые значения
    dataset = pca.transform(dataset)
    #Делим набор данных на тренировочные и тестовые согласно заданию
    X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                        y_train,
                                                        test_size=0.3,
                                                        random_state=126)
    #Находим среднее 0-ого столбца
    print("Mean of 0-coloumn in train dataset: ", pd.DataFrame(X_train)[0].mean())
    #Инициализируем классификатор и тут же обучаем его на получившемся наборе данных
    clf = OneVsRestClassifier(RandomForestClassifier(criterion='gini',
                                                     min_samples_leaf=10,
                                                     max_depth=20,
                                                     n_estimators=10,
                                                     random_state=126)).fit(X_train, y_train)
    #Здесь считаем кол-во верно классифицированных объектов класса 5
    class5 = []
    for i in range(len(y_test)):
        if y_test[i] == 5:
            class5.append(i)
    counter = 0
    y_pred = clf.predict(X_test)
    for i in range(len(class5)):
        if y_pred[class5[i]] == 5:
            counter += 1
    print("Correctly classified for class 5: ", counter)
    #Импортируем новый набор данных
    dataset2 = pd.read_csv('pred_for_task.csv', index_col='FileName')
    #Отсекаем отклик
    x_new = np.array(dataset2.drop(['Label'], axis=1))
    #Новые значения для нового набора данных согласно ранее обученному МГК
    x_new = pca.transform(x_new)
    #Новые классы согласно ранее обученному классификатору
    y_new = clf.predict(x_new)
    #Выводим ответы
    print("Predicted class for file20: ", y_new[19])
    print("Probability of Class", y_new[19], "for file20: ", clf.predict_proba(x_new)[19][y_new[19]])


if __name__ == '__main__':
    main()

