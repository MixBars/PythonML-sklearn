from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def main():
    #Part 1
    dataset = pd.read_csv('data.csv', index_col='id')
    X = np.array(dataset.X).reshape(-1, 1)
    Y = np.array(dataset.Y)
    print("Sample mean X: ", X.mean(axis=0))
    print("Sample mean Y: ", Y.mean(axis=0))
    model = LinearRegression().fit(X, Y)
    print("Theta(0): ", model.intercept_)
    print("Theta(1): ", model.coef_)
    print("R^2 statistics: ", model.score(X, Y))

    #Part 2
    candys = pd.read_csv('candy-data.csv', index_col='competitorname')
    dum = np.array(candys.loc['Dum Dums'].drop(['winpercent', 'Y'])).reshape(1, -1)
    nestle = np.array(candys.loc['Nestle Smarties'].drop(['winpercent', 'Y'])).reshape(1, -1)
    random = np.array([1,1,1,1,0,1,0,0,0,0.128,0.37]).reshape(1, -1)

    candys = candys.drop(['Dum Dums', 'Nestle Smarties'], axis = 0)
    X = np.array(candys.drop(['winpercent', 'Y'], axis=1))
    Y = np.array(candys.winpercent)
    model = LinearRegression().fit(X, Y)
    print("Predict for candy 'Dum Dums': ", model.predict(dum))
    print("Predict for candy 'Nestle Smarties': ", model.predict(nestle))
    print("Predict for special candy: ", model.predict(random))

if __name__ == '__main__':
    main()
