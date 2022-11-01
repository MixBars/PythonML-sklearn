import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def main():
    # Сохраняем данные из файла
    dataset = pd.read_csv('diabetes.csv')

    # Берем первые 620 строк
    dataset = dataset.iloc[:620]

    # считаем кол-во людей без диабета
    counter = 0
    for i in range(len(dataset)):
        if dataset.iloc[i].Outcome == 0:
            counter += 1
    print("Count of patients without diabetes in dataset: ", counter)

    # Разделяем набор данных на предикторы и на отклики
    x = dataset.drop(['Outcome'], axis=1)
    y = dataset.Outcome

    # Разделяем получившиеся наборы данных на тренировочный и тестовые
    X_train = x[:496]
    X_test = x[496:]
    y_train = y[:496]
    y_test = y[496:]

    # Обучаем машину методом дерева принятия решений
    clf = DecisionTreeClassifier(criterion='entropy',
                                 min_samples_leaf=10,
                                 max_leaf_nodes=15,
                                 random_state=2020)
    clf.fit(X_train, y_train)

    # Экспортируем дерево принятия решений
    columns = list(X_train.columns)
    export_graphviz(clf, out_file='tree.dot',
                    feature_names=columns,
                    class_names=['0', '1'],
                    rounded=True, proportion=False,
                    precision=2, filled=True, label='all')

    with open('tree.dot') as f:
        dot_graph = f.read()

    # Сохраняем получившееся дерево в формате PNG
    graph = graphviz.Source(dot_graph, format="png")
    graph.render("decision_tree_graphivz")

    # Предсказываем результаты на тестовом наборе данных
    y_predicted = clf.predict(X_test)

    # Высчитываем среднее значение метрики F1
    f1 = f1_score(y_test, y_predicted, average=None)
    print("F1 score: ", ((f1[0] + f1[1]) / 2))

    # Высчитываем дулю правильных ответов (метрика Accuracy)
    accuracy = accuracy_score(y_test, y_predicted)
    print("Accuracy score: ", accuracy)

    # Еще раз сохраняем данные из файла
    dataset2 = pd.read_csv('diabetes.csv').drop(['Outcome'], axis=1)

    # Предсказываем значение отклика с помощью обученной машины
    new_predict = clf.predict(dataset2)

    # Выводим данные для конкретных пациентов
    print("Outcome predict for pacient №754: ", new_predict[754])
    print("Outcome predict for pacient №756: ", new_predict[756])
    print("Outcome predict for pacient №759: ", new_predict[759])
    print("Outcome predict for pacient №721: ", new_predict[721])


if __name__ == '__main__':
    main()

