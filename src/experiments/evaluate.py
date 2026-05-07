# src/experiments/evaluate.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_svm(K_train, y_train, K_test, y_test, C=1.0):
    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    return acc