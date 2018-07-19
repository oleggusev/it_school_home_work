import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

# for memory estimation
# import sys
# from memory_profiler import profile

class Learning():
    # query is <aerospike.Query object>
    # need call query.results() to get data from DB
    query_feature = []

    label_name = None
    label_name_hash = None

    # DataFrame objects of query result
    data_feature = []
    data_label = []

    # Debug object
    parent = {}

    classifier = {}

    allowed_features = []

    # 0 = no
    # 1 = yes
    # None = we do not know the result, cuz logic was not started
    # defined in Customer parent class
    is_enough_data_for_dummy = None

    estimation = None

    y_test = None

    def __init__(self):
        # reset if class was in cycle
        self.data_feature = []
        self.data_label = []
        self.parent = {}
        self.classifier = {}
        self.estimation = None

        #self.classifier = RandomForestClassifier()
        self.classifier = LogisticRegression()

    # python3 -m memory_profiler avatar/coefficients.py
    # @profile
    def run(self):
        if not self.query_feature or not self.allowed_features or self.is_enough_data_for_dummy is None:
            print('Learning: error - no data for ML')
            return 0

        self.prepare_data_frame()
        # split data for test AND train
        train, test = self.stratified_split(self.data_label)

        X_train = self.data_feature.iloc[train] # 80%
        X_test = self.data_feature.iloc[test]   # 20%

        y_train = self.data_label[train]
        self.y_test = self.data_label[test]

        if (not sum(y_train)):
            # no any positive class in train data
            return 0.0
        # model learn on train data
        self.classifier.fit(X_train, y_train)
        # model takes real data and do prediction on 20% of test data
        self.y_pred = self.classifier.predict(X_test)

        self.estimation = self.balanced_classification_rate(self.y_test, self.y_pred)

        return self.estimation


    # TODO: potential issue with performance memory !!! How to divide data and learn it by peaces?
    # TODO: pandas object: 2500 rows takes 25Mb!!!
    # should I use pandas here?
    def prepare_data_frame(self):
        self.data_feature = []
        # self.query.results() list from 2525 rows x 2000 columns takes 21Kb - it's ok
        for (key, metadata, bins) in self.query_feature.results():
            row = self.remove_not_allowed_features(bins)
            self.data_feature.append(row)
        # self.parent.log(sys.getsizeof(self.data))
        # self.parent.log(self.data.memory_usage(deep=True))
        self.data_feature = pd.DataFrame(self.data_feature)

        self.data_label = self.data_feature[self.label_name_hash]
        self.data_feature.drop([self.label_name_hash], axis=1, inplace=True)

        if not self.is_enough_data_for_dummy:
            # hash all data, cuz we do not use DUMMY
            self.data_feature = self.data_feature.applymap(hash)

        return self.data_feature

    def remove_not_allowed_features(self, row):
        result = row.copy()
        for column in row:
            if not column in self.allowed_features:
                del result[column]

        return result

    def stratified_split(self, y, proportion=0.8):
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

    # @see http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    # @see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4020838/
    # (TP + TN) / (TP + TN + FP + FN) * 100
    def accuracy_lib(self, y_test, y_pred):
        try:
            return metrics.accuracy_score(y_test, y_pred)
        except Exception as e:
            print(e)
            return 0.0

    # apply BCR method when class = 1 just <=33% in result
    # @see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4020838/
    # sensitivity/recall = TP / (TP + FN) * 100
    # specificity        = TN / (TN + TP) * 100
    # @see http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    #
    # The confusion matrix is of the following form:
    #
    #                       Predicted Positives     Predicted Negatives
    #                       -------------------     --------------------
    # True Positives	|   True Positives (TP)	    False Negatives (FN)
    # True Negatives	|   False Positives (FP)	True Negatives (TN)
    def balanced_classification_rate(self, y_test, y_pred):
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + tp)
        balanced_classification_rate = (sensitivity + specificity) / 2
        return balanced_classification_rate
