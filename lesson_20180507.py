import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

#import scipy

# https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
df = pd.read_csv('data/india/pima-indians-diabetes.data.csv')

df.columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Pregnancies - Number of times pregnant - Numeric
# Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test - Numeric
# BloodPressure - Diastolic blood pressure (mm Hg) - Numeric
# SkinThickness - Triceps skin fold thickness (mm) - Numeric
# Insulin - 2-Hour serum insulin (mu U/ml) - Numeric
# BMI - Body mass index (weight in kg/(height in m)^2) - Numeric
# DiabetesPedigreeFunction - Diabetes pedigree function - Numeric
# Age - Age (years) - Numeric
# Outcome - Class variable (0 or 1) - Numeric

print(df)

# print(df.mean(), df.median())


def norm_arr(column):
    # TODO: Roman, please, check the logic for median rate, ok?
    if abs(column.mean() - column.median()) / column.mean() > 0.5:
        mean = column.median()
    else:
        mean = column.mean()

    std = column.std()
    normalized = (column - mean) / std

    return normalized


# apply method to all columns and rows!!!
# result = df.apply(norm_arr)
# print(result)

# other way: output normalized data
# for feature in df.columns:
#     nrm = norm_arr(df[feature])
#     print(feature)
#     print(nrm.mean(), nrm.std())
#     # print(' ')
#     # print(feature + ' mean')
#     # print(nrm.mean())
#     # print(feature + ' std')
#     # print(nrm.std())


# copy array and do normalization
def norm_df(df):
    df_copied = df.copy()

    for column in df.columns:
        df_copied[column] = norm_arr(df_copied[column])

    return df_copied


df_std = norm_df(df)

print('= First normalization step =====================')
# print(df_std)

print('= Second normalization step =====================')
# TODO: Roman, please, see how to filer a (above 2.5%) + (below 2.5%) values, ok?
high = df_std.quantile(.975)
low = df_std.quantile(.025)
df_std = df_std[df_std <= high]
df_std = df_std[df_std >= low]
# print(df_std)

# TODO: Roman, replaced NaN By '0', cuz ZERO is AVG for each column
df_std = df_std.fillna(0)
print(df_std)

# print('111111111111111111111111111111111')
# for feature in df.columns:
#     print('')
#     print(feature)
#     # before
#     print(df[feature].mean(), df[feature].std())
#     # after normalization
#     print(df_std[feature].mean(), df_std[feature].std())


print('')
print('Graphics output')


def plot():
    plt.figure(figsize=(8, 6))

    plt.scatter(df['preg'], df['mass'],
                color='green', label='input scale', alpha=0.5)

    plt.scatter(df_std['preg'], df_std['mass'], color='red',
                label='standardized', alpha=0.3)

    plt.title('Plasma and Insulin values of the diabetes dataset')
    plt.xlabel('Pregnancy No.')
    plt.ylabel('Body Mass')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()

    plt.show()


# plot()

def plot_class():
    fig, ax = plt.subplots(2, figsize=(6,14))

    for a,d,l in zip(range(len(ax)),
               (df[['plas', 'test']].values, df_std[['plas', 'test']].values),
               ('Input scale',
                'Standardized')
                ):
        for i,c in zip(range(0,2), ('red', 'green')):
            ax[a].scatter(d[df['class'].values == i, 0],
                  d[df['class'].values == i, 1],
                  alpha=0.5,
                  color=c,
                  label='Class %s' %i
                  )
        ax[a].set_title(l)
        ax[a].set_xlabel('Plasma')
        ax[a].set_ylabel('Insulin')
        ax[a].legend(loc='upper left')
        ax[a].grid()

    plt.tight_layout()
    plt.show()


# plot_class()

#print(df.mean(), df.median())

# HOME WORK =======================================================================================================
# разбить выборку на 80/20 % в каждом классе = 1 или 0
# msk = np.random.rand(len(df['mass'])) < 0.8
# #
# # train = df['mass'][msk]
# # test = df['mass'][~msk]

print('')
print('Divide data to test and training sets')

yes = df[df['class'] == 1]
# print(yes)
no = df[df['class'] == 0]
# print(no)

msk_yes = np.random.rand(len(yes)) < 0.8
# 80 % of yes
# print(yes[msk_yes].count())
# 20# of yes
# print(yes[~msk_yes].count())

msk_no = np.random.rand(len(no)) < 0.8
# 80 % of no
# print(no[msk_no].count())
# 20# of no
# print(no[~msk_no].count())

# Unite arrays to Train
train_source = [yes[msk_yes], no[msk_no]]
train_source = pd.concat(train_source)
# Filter normalized array by source for Train data set
train = df_std.ix[train_source.index]
print('Train\'s set count = ' + str(train['preg'].count()))
print(train)

# Unite arrays to Test
test_source = [yes[~msk_yes], no[~msk_no]]
test_source = pd.concat(test_source)
# Filter normalized array by source for Test data set
test = df_std.ix[test_source.index]
print('Test\'s set count =  ' + str(test['preg'].count()))
print(test)





