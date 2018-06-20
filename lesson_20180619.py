import pandas as pd
import numpy as np

# data set taken from
# http://archive.ics.uci.edu/ml/datasets/banknote+authentication
# Attribute Information: дисперсия; косые; эксцесс; энтропия
#
# 1. variance of Wavelet Transformed image (continuous)
# 2. skewness of Wavelet Transformed image (continuous)
# 3. curtosis of Wavelet Transformed image (continuous)
# 4. entropy of image (continuous)
# 5. class (integer)
df = pd.read_csv('data/banknote/banknotes.csv')

print(df)


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            print('X%d < %.3f Gini=%.3f' % ((index + 1), row[index], gini))
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# def get_split(dataset):
#     class_values = list(set(row[-1] for row in dataset))
#     b_index, b_value, b_score, b_groups = 999, 999, 999, None
#     for index in range(len(dataset[0])-1):
#         for row in dataset:
#             groups = test_split(index, row[index], dataset)
#             gini = gini_index(groups, class_values)
#             if gini < b_score:
#                 b_index, b_value, b_score, b_groups = index, row[index], gini, groups
#     return {'index':b_index, 'value':b_value, 'groups':b_groups}
