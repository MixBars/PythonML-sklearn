import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import cv2


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
            y.append(0)
        else:
            y.append(1)
        counter -= 1

    clf = LinearSVC(C=1.35, random_state=15)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=15)

    clf.fit(X_train, y_train)
    print("Theta(370): ", clf.coef_[0][369]) # T(369)
    print("Theta(271): ", clf.coef_[0][270]) # T(270)
    print("Theta(56): ", clf.coef_[0][55]) # T(55)
    y_predicted = clf.predict(X_test)

    f1 = f1_score(y_test, y_predicted, average=None)
    print("Average Macro-F1 score: ", ((f1[0] + f1[1])/2))

    x_new = []
    imagePaths2 = sorted(list(paths.list_images('test')))
    for image in imagePaths2:
        x_new.append(extract_histogram(cv2.imread(image)))
    x_new = np.array(x_new)

    y_new = clf.predict(x_new)
    print("Predicted class for cat.1046.jpg: ", y_new[46])
    print("Predicted class for cat.1036.jpg: ", y_new[36])
    print("Predicted class for cat.1007.jpg: ", y_new[57])
    print("Predicted class for cat.1034.jpg: ", y_new[84])


if __name__ == '__main__':
    main()
