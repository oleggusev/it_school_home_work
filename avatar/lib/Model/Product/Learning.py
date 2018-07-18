import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# for memory estimation
# import sys
# from memory_profiler import profile

class Learning():
    # query is <aerospike.Query object>
    # need call query.results() to get data from DB
    query_feature = []

    label_name = None
    label_name_hash = None

    # DataFarame object of query result
    data_feature = []
    data_label = []

    # Debug object
    parent = {}

    classifier = {}

    estimation = None


    def __init__(self):
        if self.query_feature:
            print('Learning: cannot init object - no data')

        # reset all
        query_feature = []
        label_name = None
        label_name_hash = None
        data_feature = []
        data_label = []
        parent = {}
        classifier = {}
        estimation = None

        #self.classifier = RandomForestClassifier()
        self.classifier = LogisticRegression()


    # python3 -m memory_profiler avatar/coefficients.py
    # @profile
    def run(self):
        self.prepare_data_frame()
        # split data for test AND train
        train, test = self.stratified_split(self.data_label)

        X_train = self.data_feature.iloc[train] # 80%
        X_test = self.data_feature.iloc[test]   # 20%

        y_train = self.data_label[train]
        y_test = self.data_label[test]

        # model learn on train merchant
        self.classifier.fit(X_train, y_train)
        # model takes real merchant and do prediction on 20% of test merchant
        y_pred = self.classifier.predict(X_test)

        self.estimation = self.accuracy(y_test, y_pred)

        return self.estimation


    # TODO: potential issue with performance memory !!! How to divide data and learn it by peaces?
    # TODO: pandas object: 2500 rows takes 25Mb!!!
    # should I use pandas here?
    def prepare_data_frame(self):
        # self.query.results() list from 2525 rows x 2000 columns takes 21Kb - it's ok
        for (key, metadata, bins) in self.query_feature.results():
            row = self.remove_not_allowed_features(bins)
            self.data_feature.append(row)
        # self.parent.log(sys.getsizeof(self.data))
        # self.parent.log(self.data.memory_usage(deep=True))
        self.data_feature = pd.DataFrame(self.data_feature)

        self.data_label = self.data_feature[self.label_name_hash]
        self.data_feature.drop([self.label_name_hash], axis=1, inplace=True)

        return self.data_feature


    def remove_not_allowed_features(self, row):
        allowed_features = self.get_alllowed_features()
        result = row.copy()
        for column in row:
            if not column in allowed_features:
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


    def accuracy(self, y_test, y_pred):
        return 1 - sum(abs(y_test - y_pred) / len(y_test))