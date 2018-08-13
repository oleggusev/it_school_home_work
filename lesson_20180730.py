import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from ggplot import *

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")

X = mnist.data / 255.0
y = mnist.target

print(X.shape, y.shape)

feat_cols = ['pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X, columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: int(i))

X, y = None, None

print('Size of the dataframe: {}'.format(df.shape))


rndperm = np.random.permutation(df.shape[0])




# plt.gray()
# fig = plt.figure( figsize=(16,7) )
# for i in range(0,30):
#     ax = fig.add_subplot(3,10,i+1, title='Digit: ' + str(df.loc[rndperm[i],'label']) )
#     ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
#
# plt.show()

# кластеризация в тупую наших данных
# kmeans = KMeans(n_clusters=10)
# # Fitting the input data
# kmeans = kmeans.fit(df[feat_cols].loc[rndperm[0:10000]])
# labels = kmeans.predict(df[feat_cols].loc[rndperm[10001:12000]])
#
# print(sum(labels==df['label'].loc[rndperm[10001:12000]])/2000)


pca = PCA(n_components=40)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

df['label_char'] = df['label'].apply(lambda i: str(i))

# chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-three', color='label_char') ) \
#         + geom_point(size=75,alpha=0.8) \
#         + ggtitle("First and Second Principal Components colored by digit")
# chart.show()

# кластеризация в тупую наших данных
kmeans = KMeans(n_clusters=10)
# Fitting the input data
kmeans = kmeans.fit(pca_result.loc[rndperm[0:10000]])
labels = kmeans.predict(pca_result.loc[rndperm[10001:12000]])

print(sum(labels==df.loc[rndperm[10001:12000]])/2000)