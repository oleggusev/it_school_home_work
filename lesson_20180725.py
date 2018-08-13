import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# https://www.kaggle.com/c/titanic/data
data = pd.read_csv("data/titanic/train.csv")

# Convert categorical variable to numeric
data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,np.where(data["Embarked"]=="C",1, np.where(data["Embarked"]=="Q",2,3)))


# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex
# Age	Age in years
# sibsp	# of siblings / spouses aboard the Titanic
# parch	# of parents / children aboard the Titanic
# ticket	Ticket number
# fare	Passenger fare
# cabin	Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


# Cleaning dataset of NaN
data=data[[
    "Survived",
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]].dropna(axis=0, how='any')


X_train, X_test = train_test_split(data, test_size=0.3, random_state=int(time.time()))



mean_fare_survived = np.mean(X_train[X_train["Survived"]==1]["Fare"])
std_fare_survived = np.std(X_train[X_train["Survived"]==1]["Fare"])
mean_fare_not_survived = np.mean(X_train[X_train["Survived"]==0]["Fare"])
std_fare_not_survived = np.std(X_train[X_train["Survived"]==0]["Fare"])

# mean_fare_survived = 52.88
print("mean_fare_survived = {:03.2f}".format(mean_fare_survived))
# std_fare_survived = 69.86
print("std_fare_survived = {:03.2f}".format(std_fare_survived))
# mean_fare_not_survived = 22.02
print("mean_fare_not_survived = {:03.2f}".format(mean_fare_not_survived))
# std_fare_not_survived = 29.64
print("std_fare_not_survived = {:03.2f}".format(std_fare_not_survived))



# ============================================================================

def gaussian(value, mu, sigma):
    res = 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(value-mu)**2/(2*sigma**2))
    return res



# gaussian(v, mean_fare_not_survived, std_fare_not_survived)

# 0.3963987804884547
print(gaussian(67, mean_fare_not_survived, std_fare_not_survived) * 100)
# 0.5568531229217372
print(gaussian(67, mean_fare_survived, std_fare_survived) * 100)


# ============================================================================

y_pred = []
for v in X_test["Fare"].values:
    if gaussian(v, mean_fare_not_survived, std_fare_not_survived) <= gaussian(v, mean_fare_survived, std_fare_survived):
        y_pred.append(1)
    else:
        y_pred.append(0)

y_pred = np.array(y_pred)

print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))



# ============================================================================

gnb = GaussianNB()
used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("GaussianNB: Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))

# ============================================================================

gnb = BernoulliNB()
used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("BernoulliNB: Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))

# ============================================================================

gnb = MultinomialNB()
used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp", # count parents
    "Parch", # count children
    "Fare",
    "Embarked_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("MultinomialNB: Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))

# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================


from sklearn import svm

# http://scikit-learn.org/stable/modules/svm.html#svm-kernels
model = svm.SVC(C=1, gamma=1)
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more
#  about it in next section.Train the model using the training sets and check score
model.fit(X_train[used_features].values, X_train["Survived"])
model.score(X_train[used_features].values, X_train["Survived"])

y_pred = model.predict(X_test[used_features].values)

print(sum(y_pred) / sum(X_test["Survived"].values))


from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics

forest = svm.SVC(C=1, gamma=1)
forest_params = {'C': [1, 100, 1000], 'gamma': [0.0001, 0.001, 0.1]}
forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=5, verbose=True)
forest_grid.fit(X_train[used_features].values, X_train["Survived"])
forest_grid.score(X_train[used_features].values, X_train["Survived"])
print('best params: ' + str(forest_grid.best_params_))
print('best accuracy score: ' + str(forest_grid.best_score_))
forest_pred = forest_grid.predict(X_test[used_features].values)
print('accuracy score: ' + str(metrics.accuracy_score(X_test["Survived"].values, forest_pred)))


# ============================================================================
# ============================================================================
print('# ============================================================================')
# залазить в границы глубже...
# nu support vector machine

# KERNEL:
# kernel='linear'
# polynomial
# rbf
# sigmoid
model = svm.NuSVC(gamma=0.01, kernel='linear')
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more
# about it in next section.Train the model using the training sets and check score
model.fit(X_train[used_features].values, X_train["Survived"])
model.score(X_train[used_features].values, X_train["Survived"])

y_pred = model.predict(X_test[used_features].values)

# NuSVC linear: Number of mislabeled points out of a total 215 points : 48, performance 77.67%
print("NuSVC linear: Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))


model = svm.NuSVC(gamma=0.01, kernel='poly')
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more
# about it in next section.Train the model using the training sets and check score
model.fit(X_train[used_features].values, X_train["Survived"])
model.score(X_train[used_features].values, X_train["Survived"])

y_pred = model.predict(X_test[used_features].values)

# NuSVC poly: Number of mislabeled points out of a total 215 points : 57, performance 73.49%
print("NuSVC poly: Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))


model = svm.NuSVC(gamma=0.01, kernel='rbf')
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more
# about it in next section.Train the model using the training sets and check score
model.fit(X_train[used_features].values, X_train["Survived"])
model.score(X_train[used_features].values, X_train["Survived"])

y_pred = model.predict(X_test[used_features].values)

# NuSVC rbf: Number of mislabeled points out of a total 215 points : 57, performance 78%
print("NuSVC rbf: Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))


model = svm.NuSVC(gamma=0.01, kernel='sigmoid')
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more
# about it in next section.Train the model using the training sets and check score
model.fit(X_train[used_features].values, X_train["Survived"])
model.score(X_train[used_features].values, X_train["Survived"])

y_pred = model.predict(X_test[used_features].values)

# NuSVC sigmoid: Number of mislabeled points out of a total 215 points : 100, performance 53.49%
print("NuSVC sigmoid: Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))


# ============================================================================
# ============================================================================


