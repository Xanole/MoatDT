import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import *
# from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.naive_bayes import GaussianNB as NB

np.set_printoptions(threshold=np.inf)

MOAT = 1
NORMAL = 0

model_path_DT = '.\\DT.pkl'
model_path_KNN = '.\\KNN.pkl'
model_path_NB = '.\\NB.pkl'

Moat_data = '.\\Moat_train.csv'
normal_data = '.\\Normal_train.csv'


def get_data():
    data_m = pd.read_csv(Moat_data, header=None, nrows=1026)
    label_m = pd.Series([MOAT for i in range(1026)])

    data_n = pd.read_csv(normal_data, header=None, nrows=5000)
    label_n = pd.Series([NORMAL for i in range(5000)])

    X = pd.concat([data_m, data_n], axis=0, join='outer')
    Y = pd.concat([label_m, label_n], axis=0, join='outer')

    return X, Y

def train_model(X, Y, model_name):
    """
    train machine learning algorithm
    :param DataFrame X: the matrix of the entire data
    :param Series Y: the vector of the entire labels
    :param str model_name: the classification model (DT, NB or KNN)
    """
    if model_name == "DT":
        model = DT()
        model_path = model_path_DT
    elif model_name == "KNN":
        model = KNN()
        model_path = model_path_KNN
    elif model_name == "NB":
        model = NB()
        model_path = model_path_NB

    min_score = 0
    accuracy_list = []
    tpr_list = []
    fpr_list = []
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        # print(np.shape(X_train))
        # print(np.shape(X_test))
        model.fit(X_train, Y_train)

        score = model.score(X_test, Y_test)
        accuracy_list.append(score)

        result = get_stat(list(Y_test), model.predict(X_test).tolist())
        tpr_list.append(result[0])
        fpr_list.append(result[1])

        if score > min_score:
            min_score = score
            joblib.dump(model, model_path)
            # print(score)

    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    avg_tpr = sum(tpr_list) / len(tpr_list)
    avg_fpr = sum(fpr_list) / len(fpr_list)
    return avg_accuracy, avg_tpr, avg_fpr


def test_model(X, Y, model_name):
    """
    load and test machine learning algorithm
    :param DataFrame X: the matrix of the entire data
    :param Series Y: the vector of the entire labels
    :param str model_name: the classification model (DT, NB or KNN)
    """
    if model_name == "DT":
        model_path = model_path_DT
    elif model_name == "KNN":
        model_path = model_path_KNN
    elif model_name == "NB":
        model_path = model_path_NB

    model = joblib.load(model_path)
    result = get_stat(list(Y), model.predict(X).tolist())

    return result

    # result1 = cross_val_score(model, X, Y, cv=10)
    # print(result1)
    # print(result1.mean())


def get_stat(ytest, ypred):
    """
    calculate the TPR/FPR/FNR/TNR/PR_AUC
    :param list ytest: the array for the labels of the test instances
    :param list ypred: the array for the predicted labels of the
    test instances
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    label = MOAT
    for i in range(len(ypred)):
        # print(ypred[i], ytest[i])
        if ypred[i] == label and ytest[i] == label:
            tp += 1
        elif ypred[i] == label and ytest[i] != label:
            fp += 1
        elif ypred[i] != label and ytest[i] == label:
            fn += 1
        elif ypred[i] != label and ytest[i] != label:
            tn += 1
    tpr = float(tp / list(ytest).count(MOAT))
    fpr = float(fp / list(ytest).count(NORMAL))
    # fnr = float(fn / list(ytest).count(MOAT))
    # tnr = float(tn / list(ytest).count(NORMAL))
    accuracy = float((tp + tn) / (list(ytest).count(MOAT) + list(ytest).count(NORMAL)))
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = float(tp / (tp + fp))

    return [tpr, fpr, accuracy, precision]


if __name__ == '__main__':

    X, Y = get_data()

    for model_name in ["DT", "KNN", "NB"]:
        avg_accuracy, avg_tpr, avg_fpr = train_model(X, Y, model_name)
        print(model_name, round(avg_accuracy*100, 2), round(avg_tpr*100, 2), round(avg_fpr*100, 2))


    # print()
    # for model_name in ["DT", "KNN", "NB"]:
    #     result = test_model(X, Y, model_name)
    #     print(model_name, result)

