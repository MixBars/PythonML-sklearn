from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


def main():
    dataset = pd.read_csv('candy-data.csv', index_col='competitorname').drop(['winpercent'], axis=1)
    dataset = dataset.drop(['3 Musketeers', 'Almond Joy', 'Jawbusters'], axis=0)
    x_train = np.array(dataset.drop(['Y'], axis=1))
    y_train = np.array(dataset.Y)

    datatotest = pd.read_csv('candy-test.csv', index_col='competitorname')
    x_test = datatotest.drop(['Y'], axis=1)
    y_test = datatotest.Y

    tootsiePop = np.array(x_test.loc['Tootsie Pop']).reshape(1, -1)
    werthersOriginal = np.array(x_test.loc['Werthers Original Caramel']).reshape(1, -1)

    model = LogisticRegression(random_state=2019, solver='lbfgs')
    model.fit(x_train, y_train)

    y_predicted = model.predict(x_test)

    print("Probability of classifying Tootsie Pop as Class 1:", model.predict_proba(tootsiePop))
    print("Probability of classifying Werthers Original as Class 1", model.predict_proba(werthersOriginal))

    print("Precision score: ", precision_score(y_test, y_predicted))
    print("Recall score: ", recall_score(y_test, y_predicted))
    print("AUC score: ", roc_auc_score(y_test, model.decision_function(x_test)))


if __name__ == '__main__':
    main()
