import pandas as pd
import numpy as np
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
df = pd.read_csv('merchant/india/pima-indians-diabetes.merchant.csv', names=names)

# 1) откинуть 2.5% max и min и откинуть 20% тестовых данных
# (можно вообще не разбивать вначале, если данных много, а сделать это в конце)
#
# 2) посчитать среднее и медиану  на  80% данных, если их мало
#
# 3) нормализация исходного дата сета с определённой median и  mean
#
# перебор моделей регресии или класификации...


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


# y = df['class']
#
# train, test = stratified_split(y)

# print('train: ')
# print(train)
# print('test: ')
# print(test)


# X_train = df.iloc[train, 0:8]
# X_test = df.iloc[test, 0:8]
#
# y_train = df['class'][train]
# y_test = df['class'][test]

logreg = LogisticRegression()
# logreg.fit(X_train, y_train)

rf = RandomForestClassifier()
# y_pred = rf.predict(X_test)
# y_pred

# y_pred = logreg.predict(X_test)

# print('y_pred')
# print(y_pred)
# print('y_test')
# print(y_test)

# Z | y_test - y_pred | / len(y_test)

# print('y_test - y_pred')
#
# print( ( 1 - sum(abs(y_test - y_pred)) / y_test.count() ) )


# def my_CN(df, logreg, nfold=10, normalized=True):
#     acc = []
#     for x in range(nfold):
#         y = df['class']
#         train, test = stratified_split(y)
#         text = 'Normalized: '
#         if normalized:
#             # takes first 8 columns
#             X_train = norm_df(df.iloc[train, 0:8])
#             X_test = norm_df(df.iloc[test, 0:8])
#         else:
#             X_train = df.iloc[train, 0:8]
#             X_test = df.iloc[test, 0:8]
#             text = 'Source: '
#
#         y_train = df['class'][train]
#         y_test = df['class'][test]
#
#         logreg.fit(X_train, y_train)
#         y_pred = logreg.predict(X_test)
#
#         percent = (1 - sum(abs(y_test - y_pred)) / y_test.count())
#         acc.append(percent)
#
#     print(text)
#     result = np.sum(acc) / len(acc)
#     print(result)
#
#     return result
#
#
# print('=LogisticRegression: ')
# norm = my_CN(df, logreg, 100)
# not_norm = my_CN(df, logreg, 100, False)
#
#
# print('=RandomForestClassifier: ')
# norm = my_CN(df, rf, 100)
# not_norm = my_CN(df, rf, 100, False)

print('====Roman\'s method====')


def accuracy(y_test, y_pred):
    return 1 - sum(abs(y_test - y_pred)/len(y_test))


# apply BCR method when class = 1 just <=33% in result
def BCR(y, yp):
    m = y == 0
    return (accuracy(y[m], yp[m]) + accuracy(y[~m], yp[~m])) / 2


# BCR(y_test, y_pred)


# cross validation is number of cycles(folds which launch our model(classifier)
def CV(df, classifier, nfold, norm=True):
    acc = []
    for i in range(nfold):
        y = df['class']
        # split merchant set for test AND train
        train, test = stratified_split(y)

        # get normalized OR not normalized columns from merchant set
        if norm:
            X_train = norm_df(df.iloc[train, 0:8])
            X_test = norm_df(df.iloc[test, 0:8])
        else:
            X_train = df.iloc[train, 0:8]
            X_test = df.iloc[test, 0:8]

        y_train = y[train]
        y_test = y[test]

        # model learn on train merchant
        classifier.fit(X_train, y_train)
        # model takes real merchant and do prediction on 20% of test merchant
        y_pred = classifier.predict(X_test)

        # kisya = np.array([[1, 75, 60, 0, 0, 18.4, 0.47, 30]])
        # kisya = pd.DataFrame({1, 75, 60, 0, 0, 18.4, 0.47, 30})
        #kisya = np.array([])
        # kisya = np.array([[3, 120, 70, 20, 32, 31, 0.47, 30]])
        # # print(kisya)
        # y_kisya = classifier.predict(kisya)
        # print('y_kisya')
        # print(y_kisya)

        # calculated losses between train and test labels
        acc.append(accuracy(y_test, y_pred))

    return acc


fold = 100
# # LOSS calculation
# print('=LogisticRegression (yes/no): ')
# print('Norm')
# norm2 = CV(df, logreg, fold)
# print(np.sum(norm2) / len(norm2))
# print('Not norm')
# not_norm2 = CV(df, logreg, fold, False)
# print(np.sum(not_norm2) / len(not_norm2))
#
# print('=RandomForestClassifier (tree for yes/no): ')
# print('Norm')
# norm2 = CV(df, rf, fold)
# print(np.sum(norm2) / len(norm2))
# print('Not norm')
# not_norm2 = CV(df, rf, fold, False)
# print(np.sum(not_norm2) / len(not_norm2))

print(df)
norm2 = CV(df, logreg, fold)
print(np.sum(norm2) / len(norm2))
