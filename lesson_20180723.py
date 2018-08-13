# ported to python fopr stemming!
# import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import numpy as np

#twenty = fetch_20newsgroups()
#print(twenty)



twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)


#print(twenty_train.target_names)
#print("\n".join(twenty_train.data[1].split("\n")[:17]))
#print("\n".join(twenty_train.data[2000].split("\n")[:37]))

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# X_train_counts.shape
#
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_train_tfidf.shape

# text_lr = Pipeline([('vect', CountVectorizer(stop_words='english')),
#                     ('tfidf', TfidfTransformer()),
#                     ('lr', LogisticRegression(multi_class='multinomial', solver='newton-cg'))
# ])
#
# _ = text_lr.fit(twenty_train.data, twenty_train.target)
#
# predicted = text_lr.predict(twenty_test.data)
# result = np.mean(predicted == twenty_test.target)
# # 83%
# print(result)



# text_rf = Pipeline([('vect', CountVectorizer()),
#                     ('tfidf', TfidfTransformer()),
#                     ('rf', RandomForestClassifier(n_estimators=10, criterion='entropy'))
# ])
#
# _ = text_rf.fit(twenty_train.data, twenty_train.target)
#
# predicted = text_rf.predict(twenty_test.data)
# result = np.mean(predicted == twenty_test.target)
# # 36% for 10
# # 72% for 1000
# print(result)



# text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
#
# _ = text_clf.fit(twenty_train.data, twenty_train.target)
# predicted = text_clf.predict(twenty_test.data)
# result = np.mean(predicted == twenty_test.target)
# # 77%
# print(result)



# text_clf_svm = Pipeline([('vect', CountVectorizer()),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
# ])
# _ = text_clf_svm.fit(twenty_train.data, twenty_train.target)
#
# predicted = text_clf_svm.predict(twenty_test.data)
# result = np.mean(predicted == twenty_test.target)
# # 82%
# print(result)




# stemmer = SnowballStemmer("english")
#
# class StemmedCountVectorizer(CountVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#         return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
#
#
# stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
# text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
#                              ('tfidf', TfidfTransformer()),
#                              # ('mnb', MultinomialNB(fit_prior=False))
#                              # ('lr', LogisticRegression(multi_class='multinomial', solver='newton-cg'))
#                              # ('rf', RandomForestClassifier(n_estimators=100, criterion='entropy'))
#                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
#                              ])
#
# text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
# predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
# result = np.mean(predicted_mnb_stemmed == twenty_test.target)
# # with stemming:
# # MultinomialNB: 82%
# # LogisticRegression: 83%
# # RandomForestClassifier: 72%
# # SGDClassifier: 81%
# print(result)





text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)

predicted = text_clf_svm.predict(twenty_test.data)

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf-svm__alpha': (1e-2, 1e-3)}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)
# 90%
print(gs_clf_svm.best_score_)
# {
#     'clf-svm__alpha': 0.001,
#     'tfidf__use_idf': True,
#     'vect__ngram_range': (1, 2)
# }
print(gs_clf_svm.best_params_)

