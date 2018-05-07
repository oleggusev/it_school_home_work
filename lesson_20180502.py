import pandas as pd
import numpy as np

# GIT: https://github.com/oleggusev/it_school_home_work.git
# https://archive.ics.uci.edu/ml/datasets/Automobile


# Коллеги, у меня к вам следующее домашнее задание:
# 1. Прочитайте базовый курс статистики [в комиксах], который я вам скину. Несмотря на несколько примитивный язык,
# его писал прекрасный специалист в статистике, и книга даёт отличное представление о том, как нужно думать о данных.
# 2. Попрактикуйте нормализацию, и по желанию визуализацию данных следуя вот этой статье http://sebastianraschka.com/Articles/2014_about_feature_scaling.html .
# После предыдущей книги, вам будет намного проще её читать.
# 3. В качестве практики реализуйте функции нормализации, которой на вход подаётся вектор чисел, а на выходе получаем нормализованный вектор.
# Усложните функцию, добавив порог в процентах, за которым мы считаем данные выбросами (например верхние и нижние 2.5% выборки)
# и не учитываем для расчёта средних значений
#






df = pd.read_csv('data/autos/imports-85.data.csv')

df.columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration ', 'num-of-doors ', 'body-style',
                'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
                'engine-type', 'num-of-cylinders', 'engine-size ', 'fuel-system', 'bore ', 'stroke', 'compression-ratio',
                'horsepower ', 'peak-rpm', 'city-mpg ', 'highway-mpg', 'price']

# df.loc[df['price'] == '?'] = 1

print((df['price']))

# own method
def format(row):
    # filter price column
    if row == '?':
        row = 0

    row = float(row)

    return row


df['price'] = df['price'].apply(format)

# 1) формула немного неверна, не нужно брать абсолютное значение от разницы (цена - mean) – нам нужны данные которые будут колебать вокруг 0 (среднего значения).
# То есть данные нам будут говорить – если 0, то разделяющей информации такая цена не несет, она такая же как средняя по больнице.
# Если же цена сильно отличается от средней в одну или другую от 0 сторону, то она указывает на отличительную характеристику

# 2) нужно выкинуть не 5% строк сверху и снизу, а выкинуть их из рассмотрения при расчете среднего значения и средеквадратического отклонения.
# а потом точно так же применить нормализацию

normalized_price = (df['price'] - np.mean(df['price']) / df['price'].std())
print(list(normalized_price))

# quantily() http://take.ms/Kk4aw

# print(list(df['price'].apply(money)))
#
# new_df = df['price'].apply(money)
#
# print(list(new_df))



#
# df['13495'].apply(lambda row: (int(row)+10000))
# print(df['13495'])




# ============================================================================================
# continuous from 65 to 256
# df.loc[df['price'] == '?'] = 1
# print('============================')
# # print(df['price'].to_string())
#
#

#
#
# df['price'].apply(lambda row: money(int(row)+10000))
#
# print(df['price'])

# print(df)






# normalized = df.sub(axis=2, axis=2)
#
# print(normalized)

# #print(df['normalized-losses'].to_string())
#
# print("df['normalized-losses'].mean")
# #print(df['normalized-losses'].mean)
# #print(df['normalized-losses'].std)
# first = df['normalized-losses']
# second = df['normalized-losses'].mean
# third = df['normalized-losses'].std
# standard_otklonenie = (first - second) / third


#============================================================================================


# df['num-of-doors'].replace(to_replace=['one', 'two', 'three', 'four', 'five'], value=['1', '2',  '3', '4', '5'],inplace=True)

#df.loc[df['num-of-door'] == '?'] = 0
# df.loc[df['num-of-door'] == 'two'] = 2
# df.loc[df['num-of-door'] == 'four'] = 4

#print(df)
