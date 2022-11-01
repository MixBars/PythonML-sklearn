import cv2
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from imutils import paths
from sklearn.metrics import accuracy_score
import numpy as np


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def main():
    x = []
    y = []
    imagePaths = sorted(list(paths.list_images('train')))
    counter = 500
    for image in imagePaths:
        x.append(extract_histogram(cv2.imread(image)))
        if counter > 0:
            y.append(1)
        else:
            y.append(0)
        counter -= 1

    clf1 = LinearSVC(C=1.51,
                     random_state=336)

    clf2 = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                                   min_samples_leaf=10,
                                                                   max_leaf_nodes=20,
                                                                   random_state=336),
                             n_estimators=21,
                             random_state=336)

    clf3 = RandomForestClassifier(n_estimators=21,
                                  criterion='entropy',
                                  min_samples_leaf=10,
                                  max_leaf_nodes=20,
                                  random_state=336)

    clf4 = LogisticRegression(solver='lbfgs', random_state=336)

    estimators = [('svr', clf1),
                  ('bc', clf2),
                  ('rf', clf3)]

    clf = StackingClassifier(estimators=estimators, final_estimator=clf4, cv=2)
    clf.fit(x, y)

    x_new = []
    imagePaths2 = sorted(list(paths.list_images('test')))
    for image in imagePaths2:
        x_new.append(extract_histogram(cv2.imread(image)))
    x_new = np.array(x_new)

    y_pred = clf.predict(x)
    print("Accuracy score: ", accuracy_score(y, y_pred))

    y_new = clf.predict_proba(x_new)
    print("Probability of classification image dog.1038.jpg as Class 1: ", y_new[88][1])
    print("Probability of classification image cat.1012.jpg as Class 1: ", y_new[12][1])
    print("Probability of classification image cat.1022.jpg as Class 1: ", y_new[22][1])
    print("Probability of classification image dog.1021.jpg as Class 1: ", y_new[71][1])


if __name__ == '__main__':
    main()