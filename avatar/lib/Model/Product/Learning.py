import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from IPython.display import display
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

    def __init__(self):
        if not self.parent:
            print('No Debug object for print')

    # python3 -m memory_profiler avatar/coefficients.py
    # @profile
    def run(self):
        if not self.query_feature:
            return False

        self.set_data_frame(self)



        return

    # TODO: potential issue with performance memory !!! How to divide data and learn it by peaces?
    # TODO: pandas object: 2500 rows takes 25Mb!!!
    # should I use pandas here?
    def set_data_frame(self):
        # self.query.results() list from 2525 rows x 2000 columns takes 21Kb - it's ok
        for (key, metadata, bins) in self.query_feature.results():
            self.data_feature.append(bins)
        # self.parent.log(sys.getsizeof(self.data))
        # for (key, metadata, bins) in self.query_label.results():
        #     self.data_label.append(bins)



        # self.data = pd.DataFrame(self.data[:int((len(self.data)-1)/2)])
        # self.parent.log(self.data.memory_usage(deep=True))

        self.data_feature = pd.DataFrame(self.data_feature)

        self.data_label = self.data_feature[self.label_name_hash]
        display(self.data_label.head())

        return self.data_feature