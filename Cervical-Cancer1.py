

# coding: utf-8

# In[11]:

import sklearn as sk
import numpy as np
import pandas as pd


# In[36]:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier  


# In[15]:

ccd = pd.read_csv('Cervical Cancer Data.csv', header=0)


# In[16]:

data=ccd.get_values()


# In[17]:

X=data[:,0:31]


# In[18]:

y=data[:,32]


# In[19]:

y_cancerous = y[y==1]
X_cancerous = X[y==1]
X_non_cancerous  = X[y==0]
y_non_cancerous = y[y==0]
X_cancerous_train, X_cancerous_test, y_cancerous_train, y_cancerous_test = train_test_split(X_cancerous, y_cancerous, test_size=1/7, random_state=None)
X_non_cancerous_train, X_non_cancerous_test, y_non_cancerous_train, y_non_cancerous_test = train_test_split(X_non_cancerous, y_non_cancerous, test_size=1/7, random_state=None)


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/8, random_state=None)


# In[37]:

perceptron_model_normal = Perceptron(class_weight="balanced",n_iter=1000)
perceptron_model_proportional = Perceptron()


# In[34]:

X_train_proportional = np.concatenate((X_cancerous_train,X_non_cancerous_train))
y_train_proportional = np.concatenate((y_cancerous_train,y_non_cancerous_train))


# In[35]:

perceptron_model_normal.fit(X_train,y_train)
perceptron_model_proportional.fit(X_train_proportional, y_train_proportional)

logistic_classifier = LogisticRegression(penalty='l1',class_weight="balanced")
logistic_classifier.fit(X_train,y_train)
score_cv_log = np.average(cross_val_score(logistic_classifier, X_train, y_train, cv=10))
score_cv_log_test = np.average(cross_val_score(logistic_classifier, X_test, y_test, cv=8))
prediction_log_cancerous_train = logistic_classifier.predict(X_cancerous_train)
prediction_log_cancerous_test = logistic_classifier.predict(X_cancerous_test)
pred_prob_train = logistic_classifier.predict_proba(X_train)
pred_prob_test = logistic_classifier.predict_proba(X_test)
pred_prob_train_cancerous = logistic_classifier.predict_proba(X_cancerous_train)
pred_prob_test_cancerous = logistic_classifier.predict_proba(X_cancerous_test)
pred_prob_train_non_cancerous = logistic_classifier.predict_proba(X_non_cancerous_train)
pred_prob_test_non_cancerous = logistic_classifier.predict_proba(X_non_cancerous_test)
log_coeff = logistic_classifier.coef_



linearsvc = LinearSVC(penalty="l1",class_weight="balanced",dual=False)
linearsvc.fit(X_train,y_train)
score_cv_lsvc = np.average(cross_val_score(linearsvc, X_train, y_train, cv=10))
score_cv_lsvc_test = np.average(cross_val_score(linearsvc, X_test, y_test, cv=8))
prediction_lsvc_cancerous_train = linearsvc.predict(X_cancerous_train)
prediction_lsvc_cancerous_test = linearsvc.predict(X_cancerous_test)
prediction_lsvc_non_cancerous_train = linearsvc.predict(X_non_cancerous_train)
prediction_lsvc_non_cancerous_test = linearsvc.predict(X_non_cancerous_test)
incorr_lsvc_non_cancerous_train = sum(prediction_lsvc_non_cancerous_train)
lsvc_coeff = linearsvc.coef_
lsvc_log_coeffs = np.concatenate((lsvc_coeff,log_coeff,lsvc_coeff-log_coeff),axis=0)

sgdclassifier = SGDClassifier(loss="hinge",penalty="elasticnet",class_weight="balanced")
sgdclassifier.fit(X_train,y_train)
score_cv_sgd = np.average(cross_val_score(sgdclassifier, X_train, y_train, cv=10))
score_cv_sgd_test = np.average(cross_val_score(sgdclassifier, X_test, y_test, cv=8))




from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold


estimator_log = LogisticRegression(penalty='l1',class_weight="balanced")
estimator_lsvc = LinearSVC(penalty="l1",class_weight="balanced",dual=False)
estimator_perceptron = Perceptron(class_weight="balanced",n_iter=5000)


selector_perceptron = RFECV(estimator_perceptron,cv=10,scoring="accuracy")
selector_perceptron.fit(X_train,y_train)
perceptron_features = selector_perceptron.support_
perceptron_feat_rank_rfecv = selector_perceptron.ranking_

selector_log = RFECV(estimator_log,cv=10)
selector_log.fit(X_train,y_train)
log_features = selector_log.support_

selector_lsvc = RFECV(estimator_lsvc,cv=10)
selector_lsvc.fit(X_train,y_train)
lsvc_features = selector_lsvc.support_

from sklearn.feature_selection import RFE
selector_log_rfe = RFE(estimator_log,n_features_to_select=10)
selector_log_rfe.fit(X_train,y_train)
log_rfe_num_feat = selector_log_rfe.n_features_
log_rfe_feat = selector_log_rfe.support_
log_rfe_rank_feat = selector_log_rfe.ranking_
log_rfe_feat_and_rank = np.stack((log_rfe_feat,log_rfe_rank_feat))

selector_lsvc_rfe = RFE(estimator_lsvc,n_features_to_select=10)
selector_lsvc_rfe.fit(X_train,y_train)
lsvc_rfe_num_feat = selector_lsvc_rfe.n_features_
lsvc_rfe_feat = selector_lsvc_rfe.support_
lsvc_rfe_rank_feat = selector_lsvc_rfe.ranking_

selector_percept_rfe = RFE(estimator_perceptron,n_features_to_select=10)
selector_percept_rfe.fit(X_train,y_train)
percept_rfe_num_feat = selector_percept_rfe.n_features_
percept_rfe_feat = selector_percept_rfe.support_
percept_rfe_rank_feat = selector_percept_rfe.ranking_

# In[ ]:

from sklearn.model_selection import cross_val_score


# In[ ]:

score_cv_normal = np.average(cross_val_score(perceptron_model_normal, X_train, y_train, cv=10))
score_cv_proportional = np.average(cross_val_score(perceptron_model_proportional, X_train_proportional, y_train_proportional, cv=10))
print(score_cv_normal)
print(score_cv_proportional)


# In[ ]:

p_coeff = perceptron_model_normal.coef_
perceptron_model_proportional.coef_

