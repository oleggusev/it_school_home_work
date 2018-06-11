import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Pregnancies - Number of times pregnant - Numeric
# Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test - Numeric
# BloodPressure - Diastolic blood pressure (mm Hg) - Numeric
# SkinThickness - Triceps skin fold thickness (mm) - Numeric
# Insulin - 2-Hour serum insulin (mu U/ml) - Numeric
# BMI - Body mass index (weight in kg/(height in m)^2) - Numeric
# DiabetesPedigreeFunction - Diabetes pedigree function - Numeric
# Age - Age (years) - Numeric
# Outcome - Class variable (0 or 1) - Numeric

# df = pd.read_csv(url, names=names)
df = pd.read_csv('data/india/pima-indians-diabetes.data.csv', names=names)

# df.boxplot()
#
# df.hist()

# 2 graph groups for class=0 AND class = 1
# df.groupby('class').hist()

# 2 graph groups for class=0 AND class = 1 FOR 1 column!!!
# df.groupby('mass').plas.hist(alpha=0.4)
#
# plt.show()
#
# exit()


def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized


def norm_df(df):
    result = df.copy()

    for feature in df.columns:
        result[feature] = norm_arr(result[feature])

    return result


def stratified_split(y, proportion=0.8):
    y = np.array(y)

    train_inds = np.zeros(len(y), dtype=bool)
    test_inds = np.zeros(len(y), dtype=bool)

    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y == value)[0]
        np.random.shuffle(value_inds)

        n = int(proportion * len(value_inds))

        train_inds[value_inds[:n]] = True
        test_inds[value_inds[n:]] = True

    return train_inds, test_inds


def accuracy(y_test, y_pred):
    return 1 - sum(abs(y_test - y_pred)/len(y_test))


# cross validation is number of cycles(folds which launch our model(classifier)
def CV(df, classifier, nfold, norm=True, all_columns=True):
    acc = []
    for i in range(nfold):
        y = df['class']
        # split data set for test AND train
        train, test = stratified_split(y)

        # get normalized OR not normalized columns from data set
        if norm:
            if (all_columns):
                X_train = norm_df(df.iloc[train, 0:8])
                X_test = norm_df(df.iloc[test, 0:8])
            else:
                X_train = norm_df(df.ix[train, ['age', 'plas', 'preg', 'pedi']])
                X_test = norm_df(df.ix[test, ['age', 'plas', 'preg', 'pedi']])
        else:
            if (all_columns):
                X_train = df.iloc[train, 0:8]
                X_test = df.iloc[test, 0:8]
            else:
                X_train = df.ix[train, ['age', 'plas', 'preg', 'pedi']]
                X_test = df.ix[test, ['age', 'plas', 'preg', 'pedi']]

        y_train = y[train]
        y_test = y[test]

        # model learn on train data
        classifier.fit(X_train, y_train)
        # model takes real data and do prediction on 20% of test data
        y_pred = classifier.predict(X_test)

        # calculated losses between train and test labels
        acc.append(accuracy(y_test, y_pred))

    return acc


# cross validation is number of cycles(folds which launch our model(classifier)
def CV_PLUS(df, classifier, nfold, norm=True, all_columns=True, col=[]):
    acc = []
    for i in range(nfold):
        y = df['class']
        # split data set for test AND train
        train, test = stratified_split(y)

        columns = ['age', 'plas', 'preg', 'pedi']
        # get normalized OR not normalized columns from data set
        if norm:
            if (all_columns):
                X_train = norm_df(df.iloc[train, 0:8])
                X_test = norm_df(df.iloc[test, 0:8])
            else:
                X_train = norm_df(df.ix[train, columns])
                X_test = norm_df(df.ix[test, columns])
        else:
            if (all_columns):
                X_train = df.iloc[train, 0:8]
                X_test = df.iloc[test, 0:8]
            else:
                X_train = df.ix[train, columns]
                X_test = df.ix[test, columns]
        print(X_train)
        y_train = y[train]
        y_test = y[test]

        # model learn on train data
        classifier.fit(X_train, y_train)
        # model takes real data and do prediction on 20% of test data
        y_pred = classifier.predict(X_test)

        # calculated losses between train and test labels
        acc.append(accuracy(y_test, y_pred))

    return acc


fold = 1

logreg = LogisticRegression()
rf = RandomForestClassifier()

# print('All columns')
# norm2 = CV(df, logreg, fold)
# print(np.sum(norm2) / len(norm2))
#
# norm2 = CV(df, rf, fold)
# print(np.sum(norm2) / len(norm2))
#
# print('==============')
# print('All columns without normalization')
#
# norm2 = CV(df, logreg, fold, False)
# print(np.sum(norm2) / len(norm2))
#
# norm2 = CV(df, rf, fold, False)
# print(np.sum(norm2) / len(norm2))
#
#
#
# print('==============')
# print('4 columns')
# norm2 = CV(df, logreg, fold, True, False)
# print(np.sum(norm2) / len(norm2))
#
# norm2 = CV(df, rf, fold, True, False)
# print(np.sum(norm2) / len(norm2))
#
# print('==============')
# print('4 columns without without normalization')
# norm2 = CV(df, logreg, fold, True, False)
# print(np.sum(norm2) / len(norm2))
#
# norm2 = CV(df, rf, fold, True, False)
# print(np.sum(norm2) / len(norm2))


# columns=['age', 'plas', 'preg', 'pedi']
#
# for column in columns:
#     norm2 = CV_PLUS(df, rf, fold, True, False, column)
#     print(np.sum(norm2) / len(norm2))

# home work
# + pair columns
#