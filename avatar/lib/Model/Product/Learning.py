import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
import sklearn.metrics as metrics

import math
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

    estimations = {}

    y_test = None

    entity_id = None

    entity_type = 'product'

    # LIMIT_BOUGHT_COUNT = 10
    THRESHOLD_UNIT_PERCENT = 25
    THRESHOLD_ZERO_RATE = 1.6 # heuristic

    def __init__(self):
        # reset if class was in cycle
        self.data_feature = []
        self.data_label = []
        self.parent = {}
        self.classifier = {}
        self.estimations = {}
        self.max = {}
        self.y_test = []
        self.y_pred = []

    # python3 -m memory_profiler avatar/coefficients.py
    # @profile
    def run(self):
        if not self.query_feature or not self.allowed_features or self.is_enough_data_for_dummy is None:
            self.log('Learning: error - no data for ML')
            return False

        if not len(self.prepare_data_frame()):
            return False

        self.do_cross_validation()

        return self.get_the_best_estimation()

    # if fold=10000 - 1.1 mins/product -> production mode
    # if fold=1000  - 0.2 mins/product -> developer mode
    def do_cross_validation(self, fold = 1000):
        self.estimations[self.entity_id] = pd.DataFrame(
            columns=['type', 'id', 'bcr', 'accuracy', 'predicted_y', 'coefficients']
        )
        for i in range(fold):
            # split data for test AND train
            train, test = self.stratified_split(self.data_label.copy())

            X_train = self.data_feature.iloc[train] # 80%
            X_test = self.data_feature.iloc[test]   # 20%

            y_train = self.data_label[train]
            self.y_test = self.data_label[test]

            if (not sum(y_train)):
                # no any positive class in train data
                continue
            self.classifier = LogisticRegression()
            # model learn on train data
            self.classifier.fit(X_train, y_train)
            # model takes real data and do prediction on 20% of test data
            self.y_pred = self.classifier.predict(X_test)

            self.collect_estimations()
        return

    def collect_estimations(self):
        bcr = self.balanced_classification_rate(self.y_test, self.y_pred)
        accuracy_lib = 0.0
        label_predicted = 0.0
        if len(self.y_pred) and len(self.y_test):
            accuracy_lib = round(self.accuracy_lib(self.y_test, self.y_pred) * 100, 2)
            label_predicted = round(sum(self.y_pred) / sum(self.y_test) * 100, 2)

        self.estimations[self.entity_id] = self.estimations[self.entity_id].append({
            'type': self.entity_type,
            'id': self.entity_id,
            'bcr': float(round(bcr, 2)),
            'accuracy': float(accuracy_lib),
            'predicted_y': float(label_predicted),
            'coefficients': self.classifier.coef_
        }, ignore_index=True)

    def get_the_best_estimation(self):
        estimation = self.estimations[self.entity_id][['bcr', 'accuracy']]
        estimation = estimation.copy()
        rank = estimation.rank(method='max')
        estimation['rank'] = rank.sum(axis=1)
        estimation.sort_values(by=['rank'], ascending=False, inplace=True)
        self.log('\nMAX estimation:')
        self.log(estimation.iloc[0])
        # save just the BEST coefficients for current product
        self.max[self.entity_id] = pd.DataFrame(
            columns=['type', 'id', 'bcr', 'accuracy', 'predicted_y', 'coefficients']
        )
        self.max[self.entity_id] = self.max[self.entity_id].append(
            self.estimations[self.entity_id].loc[estimation.first_valid_index()]
        , ignore_index=True)

        self.log('\nAVG estimations:')
        self.printDictionary(
            {
                'bcr': round(
                    sum(self.estimations[self.entity_id]['bcr'])
                    / len(self.estimations[self.entity_id]['bcr'])
                    , 2),
                'accuracy': round(
                    sum(self.estimations[self.entity_id]['accuracy'])
                    / len(self.estimations[self.entity_id]['accuracy'])
                    , 2),
                'predicted_y': round(
                    sum(self.estimations[self.entity_id]['predicted_y'])
                    / len(self.estimations[self.entity_id]['predicted_y'])
                    , 2),
            }
        )
        return self.balanced_classification_rate(self.y_test, self.y_pred)


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

        if not self.is_enough_data_for_dummy:
            self.reorganization_data_original()

        if len(self.data_feature) <= len(self.fields_categorized) * math.log(len(self.fields_categorized)):
            self.log('\nLearning: not enough actions for Machine Learning = ' + str(len(self.data_feature)))
            return {}

        self.data_label = self.data_feature[self.label_name_hash]
        self.data_feature.drop([self.label_name_hash], axis=1, inplace=True)

        if not self.is_enough_data_for_dummy:
            # convert text data to float, cuz no DUMMY
            self.normalization_data_original()

        return self.data_feature

    def reorganization_data_original(self):
        total = len(self.data_feature[self.label_name_hash])
        unit = sum(self.data_feature[self.label_name_hash])
        bought_percent = (unit * 100 / total)

        if bought_percent < self.THRESHOLD_UNIT_PERCENT:
            self_data_feature = self.data_feature.copy()
            threshold_zero_percent = (bought_percent * self.THRESHOLD_ZERO_RATE)

            # filter by category does not work when, cuz BCR is 50%
            # self_data_feature_categories_zero = self_data_feature[
            #     (~np.isnan(self_data_feature['category_id']))
            #     &
            #     (self_data_feature[self.label_name_hash] == 0)
            # ]
            # if len(self_data_feature_categories_zero) >= total * threshold_zero_percent / 100:
            #     # zeros with category_id(s)
            #     zero = np.array(self_data_feature_categories_zero[self.label_name_hash] == 0)
            # else:
                # random zeros
            zero = np.array(self_data_feature[self.label_name_hash] == 0)

            df_zero_all = self_data_feature.iloc[zero]
            main_percent, rest_percent = self.stratified_split(df_zero_all[self.label_name_hash], threshold_zero_percent / 100)
            df_zero_percent = df_zero_all.iloc[main_percent]
            unit_indexes = np.array(self_data_feature[self.label_name_hash] == 1)
            # merge units and zeros
            self.data_feature = pd.concat([self_data_feature.iloc[unit_indexes], df_zero_percent])
            self.data_feature.sort_index(inplace=True)


    def normalization_data_original(self):
        self.data_feature.fillna('0', inplace=True)
        self.data_feature.replace(np.nan, '0', inplace=True)
        for column in self.data_feature:
            if column in self.fields_string:
                # make dictionary from each column, and replace to IDs
                self.data_feature[column] = LabelEncoder.fit_transform(LabelEncoder, self.data_feature[column])
            #convert data column object, text -> float
            self.data_feature[column] = self.data_feature[column].astype(str).astype(float)
            if self.data_feature[column].dtype == object:
                self.data_feature[column] = self.data_feature[column].object.replace('(\D+)', '').astype(float)


        # self.data_feature.hist()
        # self_data_feature = self.data_feature.copy()

        # qt = QuantileTransformer(output_distribution='normal')
        # self.data_feature = qt.fit_transform(self.data_feature)

        # normalizer = Normalizer()
        # self_data_feature = normalizer.transform(self_data_feature)
        # self.data_feature = pd.DataFrame(self_data_feature)
        #self.data_feature.hist()

        # no sense to use it - the result is -1% or -2% as with LabelEncoder()!!!
        # for column_name in self.data_feature:
        #     column = self.data_feature[column_name]
        #
        #     if abs(column.mean() - column.median()) / column.mean() > 0.5:
        #         mean = column.median()
        #     else:
        #         mean = column.mean()
        #
        #     std = column.std()
        #     self.data_feature[column_name] = (column - mean) / std

        #self.data_feature.hist()
        #plt.show()

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
        if not tn:
            tn = 0
        if not fp:
            fp = 0
        if not fn:
            fn = 0
        if not tp:
            tp = 0
        sensitivity = tp / (tp + fn)
        if tp or tn:
            specificity = tn / (tn + tp)
        else:
            specificity = 0
        balanced_classification_rate = (sensitivity + specificity) / 2
        return balanced_classification_rate
