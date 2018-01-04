
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
data_new=ccd.get_values()


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
score_log_test = logistic_classifier.score(X_test, y_test)
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
score_lsvc_test = linearsvc.score(X_test, y_test)
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
score_sgd_test = sgdclassifier.score(X_test, y_test)




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

np.random.shuffle(data_new)
X_new_train = data_new[0:700,0:28]
X_new_test = data_new[701:858,0:28]
y_new_hinselmann_train = data_new[0:700,28]
y_new_hinselmann_test = data_new[701:858,28]
y_new_schiller_train = data_new[0:700,29]
y_new_schiller_test = data_new[701:858,29]
y_new_citology_train = data_new[0:700,30]
y_new_citology_test = data_new[701:858,30]

logistic_classifier_hinselmann = LogisticRegression(penalty='l1',class_weight="balanced")
logistic_classifier_schiller = LogisticRegression(penalty='l1',class_weight="balanced")
logistic_classifier_citology = LogisticRegression(penalty='l1',class_weight="balanced")

logistic_classifier_hinselmann.fit(X_new_train, y_new_hinselmann_train)
logistic_classifier_schiller.fit(X_new_train, y_new_schiller_train)
logistic_classifier_citology.fit(X_new_train, y_new_citology_train)

score_cv_log_hinselmann = np.average(cross_val_score(logistic_classifier_hinselmann, X_new_train, y_new_hinselmann_train, cv=10))
score_cv_log_test_hinselmann = np.average(cross_val_score(logistic_classifier_hinselmann, X_new_test, y_new_hinselmann_test, cv=3))

score_cv_log_schiller = np.average(cross_val_score(logistic_classifier_schiller, X_new_train, y_new_schiller_train, cv=10))
score_cv_log_test_schiller = np.average(cross_val_score(logistic_classifier_schiller, X_new_test, y_new_schiller_test, cv=4))

score_cv_log_citology = np.average(cross_val_score(logistic_classifier_citology, X_new_train, y_new_citology_train, cv=10))
score_cv_log_test_citology = np.average(cross_val_score(logistic_classifier_citology, X_new_test, y_new_citology_test, cv=4))

estimator_log_hinselmann = LogisticRegression(penalty='l1',class_weight="balanced")
estimator_log_schiller = LogisticRegression(penalty='l1',class_weight="balanced")
estimator_log_citology = LogisticRegression(penalty='l1',class_weight="balanced")

selector_log_hinselmann = RFECV(estimator_log_hinselmann,cv=10)
selector_log_hinselmann.fit(X_new_train,y_new_hinselmann_train)
log_features_hinselmann = selector_log_hinselmann.support_

selector_log_schiller = RFECV(estimator_log_schiller,cv=10)
selector_log_schiller.fit(X_new_train,y_new_schiller_train)
log_features_schiller = selector_log_schiller.support_


selector_log_citology = RFECV(estimator_log_citology,cv=10)
selector_log_citology.fit(X_new_train,y_new_citology_train)
log_features_citology = selector_log_citology.support_

selector_log_hinselmann_rfe = RFE(estimator_log_hinselmann,n_features_to_select=14)
selector_log_schiller_rfe = RFE(estimator_log_schiller,n_features_to_select=14)
selector_log_citology_rfe = RFE(estimator_log_citology,n_features_to_select=14)

selector_log_hinselmann_rfe.fit(X_new_train,y_new_hinselmann_train)
log_rfe_num_feat_hinselmann = selector_log_hinselmann_rfe.n_features_
log_rfe_feat_hinselmann = selector_log_hinselmann_rfe.support_
log_rfe_rank_feat_hinselmann = selector_log_hinselmann_rfe.ranking_
log_rfe_feat_and_rank_hinselmann = np.stack((log_rfe_feat_hinselmann,log_rfe_rank_feat_hinselmann))

selector_log_schiller_rfe.fit(X_new_train,y_new_schiller_train)
log_rfe_num_feat_schiller = selector_log_schiller_rfe.n_features_
log_rfe_feat_schiller = selector_log_schiller_rfe.support_
log_rfe_rank_feat_schiller = selector_log_schiller_rfe.ranking_
log_rfe_feat_and_rank_schiller = np.stack((log_rfe_feat_schiller,log_rfe_rank_feat_schiller))

selector_log_citology_rfe.fit(X_new_train,y_new_citology_train)
log_rfe_num_feat_citology = selector_log_citology_rfe.n_features_
log_rfe_feat_citology = selector_log_citology_rfe.support_
log_rfe_rank_feat_citology = selector_log_citology_rfe.ranking_
log_rfe_feat_and_rank_citology = np.stack((log_rfe_feat_citology,log_rfe_rank_feat_citology))

#%%

data_modified_features = data

np.random.shuffle(data_modified_features)
X_mod_feat_train = data_modified_features[0:700,[0,1,2,3,5,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,24,25,26,27,28]]
X_mod_feat_test = data_modified_features[701:858,[0,1,2,3,5,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,24,25,26,27,28]]
y_mod_feat_hinselmann_train = data_modified_features[0:700,29]
y_mod_feat_hinselmann_test = data_modified_features[701:858,29]
y_mod_feat_schiller_train = data_modified_features[0:700,30]
y_mod_feat_schiller_test = data_modified_features[701:858,30]
y_mod_feat_citology_train = data_modified_features[0:700,31]
y_mod_feat_citology_test = data_modified_features[701:858,31]

logistic_classifier_hinselmann_mod_feat = LogisticRegression(penalty='l1',class_weight="balanced")
logistic_classifier_schiller_mod_feat = LogisticRegression(penalty='l1',class_weight="balanced")
logistic_classifier_citology_mod_feat = LogisticRegression(penalty='l1',class_weight="balanced")

logistic_classifier_hinselmann_mod_feat.fit(X_mod_feat_train, y_mod_feat_hinselmann_train)
logistic_classifier_schiller_mod_feat.fit(X_mod_feat_train, y_mod_feat_schiller_train)
logistic_classifier_citology_mod_feat.fit(X_mod_feat_train, y_mod_feat_citology_train)

score_cv_log_hinselmann_mod_feat = np.average(cross_val_score(logistic_classifier_hinselmann_mod_feat, X_mod_feat_train, y_mod_feat_hinselmann_train, cv=10))
score_log_test_hinselmann_mod_feat = logistic_classifier_hinselmann_mod_feat.score(X_mod_feat_test, y_mod_feat_hinselmann_test)

score_cv_log_schiller_mod_feat = np.average(cross_val_score(logistic_classifier_schiller_mod_feat, X_mod_feat_train, y_mod_feat_schiller_train, cv=10))
score_log_test_schiller_mod_feat = logistic_classifier_schiller_mod_feat.score(X_mod_feat_test, y_mod_feat_schiller_test)

score_cv_log_citology_mod_feat = np.average(cross_val_score(logistic_classifier_citology_mod_feat, X_mod_feat_train, y_mod_feat_citology_train, cv=10))
score_log_test_citology_mod_feat = logistic_classifier_citology_mod_feat.score(X_mod_feat_test, y_mod_feat_citology_test)
log_coeff = logistic_classifier.coef_

#%%

cct = pd.read_csv('New Data.csv', header=0)

new_data=cct.get_values()
X_newf_test = new_data[:,0:25]
y_newf_test = new_data[:,25]
score_log_hinselmann_new_data = logistic_classifier_hinselmann_mod_feat.score(X_newf_test,y_newf_test)
score_log_test_schiller_new_data = logistic_classifier_schiller_mod_feat.score(X_newf_test,y_newf_test)
score_log_test_citology_new_data = logistic_classifier_citology_mod_feat.score(X_newf_test,y_newf_test)
