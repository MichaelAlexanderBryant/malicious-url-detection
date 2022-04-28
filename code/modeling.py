# Guessing 1:
# recall=1, precision=accuracy=0.12128, ROC AUC=0.5
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7911559/
# https://pypi.org/project/pyarc/

# I tried changing priors for naive Bayes and linear discriminant analysis,
# but was ineffective.
# Next try downsampling or upsampling. SMOTE will probably not be effective here,
# because it relies on looking at nearest neighbors and this is high dimensional
# data.

# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from xgboost import XGBClassifier
# from pyarc import CBA, TransactionDB
from sklearn.model_selection import cross_val_score
from numpy import mean, std
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import (recall_score, accuracy_score, precision_score, confusion_matrix,
                             precision_recall_curve)
from sklearn import metrics

# Load split data.
X_train = pd.read_csv('../output/imputation/X_train.csv')
y_train = pd.read_csv('../output/imputation/y_train.csv')
y_train = np.ravel(np.array(y_train))
X_test = pd.read_csv('../output/imputation/X_test.csv')
y_test = pd.read_csv('../output/imputation/y_test.csv')

lr = LogisticRegression(max_iter = 2000, class_weight = {0:1,1:1})
cv = cross_val_score(lr,X_train,y_train,cv=5, scoring='f1')
print(mean(cv), '+/-', std(cv))

# Class_weight is a form of downsampling to accomodate unbalanced dataset.
rf = RandomForestClassifier(random_state = 1, class_weight = {0:1,1:5})
cv = cross_val_score(rf,X_train,y_train,cv=5, scoring = 'f1')
print(mean(cv), '+/-', std(cv))

# Class_weight is a form of cost-sensitive training to accomodate unbalanced dataset.
# https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/
svc = SVC(probability = True, class_weight = {0:1,1:25})
cv = cross_val_score(svc,X_train,y_train,cv=5, scoring='f1')
print(mean(cv), '+/-', std(cv))

gnb = GaussianNB()
cv = cross_val_score(gnb,X_train,y_train,cv=5, scoring = 'f1')
print(mean(cv), '+/-', std(cv))

bnb = BernoulliNB()
cv = cross_val_score(bnb,X_train,y_train,cv=5, scoring = 'f1')
print(mean(cv), '+/-', std(cv))

lda = LinearDiscriminantAnalysis(priors=[0.6,0.4])
cv = cross_val_score(lda,X_train,y_train,cv=5, scoring = 'f1')
print(mean(cv), '+/-', std(cv))

xgbc = XGBClassifier(scale_pos_weight=8,use_label_encoder = False, eval_metric='error')
cv = cross_val_score(xgbc,X_train,y_train,cv=5, scoring = 'f1')
print(mean(cv), '+/-', std(cv))

# cba = CBA(support=0.20, confidence=0.5, algorithm="m1")

# Performance reporting function.
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: {} +/- {}'.format(str(classifier.best_score_),str(
        classifier.cv_results_['std_test_score'][classifier.best_index_])))
    print('Best Parameters: ' + str(classifier.best_params_))
    
lr = LogisticRegression(class_weight = {0:1,1:1})
param_grid = {'max_iter' : [15000],
              'C' : np.arange(.001,.015,.001)
             }
clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, scoring='f1',
                      n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train,y_train)
clf_performance(best_clf_lr,'Logistic Regression')
    

rf = RandomForestClassifier(random_state = 1, class_weight = {0:1,1:5})
param_grid =  {
                'n_estimators': np.arange(8,20,2), 
                'bootstrap': [True,False], #bagging (T) vs. pasting (F)
                'max_depth': [10],
                'max_features': ['auto','sqrt'],
                'min_samples_leaf': np.arange(2,6,1),
                'min_samples_split': np.arange(2,6,1)
              }
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, scoring='f1',
                          n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,y_train)
clf_performance(best_clf_rf,'RandomForestClassifier')


svc = SVC(probability = True, random_state = 1, class_weight = {0:1,1:25})
param_grid = {
              'kernel': ['linear', 'poly', 'sigmoid','rbf'],
              'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
              'C': np.arange(40,70,5)
             }
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, scoring = 'f1',
                       n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train,y_train)
clf_performance(best_clf_svc,'SVC')

gnb = GaussianNB()
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
clf_gnb = GridSearchCV(gnb, param_grid = param_grid, cv = 5, scoring = 'f1',
                       n_jobs = -1)
best_clf_gnb = clf_gnb.fit(X_train,y_train)
clf_performance(best_clf_gnb,'gaussianNB')

lda = LinearDiscriminantAnalysis()
param_grid = {'solver': ['svd']}
clf_lda = GridSearchCV(lda, param_grid = param_grid, cv = 5, scoring = 'f1',
                       n_jobs = -1)
best_clf_lda = clf_lda.fit(X_train,y_train)
clf_performance(best_clf_lda,'LinearDiscriminantAnalysis')

xgbc = XGBClassifier(eval_metric='error')
param_grid = {'learning_rate': [0.001], 'max_depth': np.arange(1,9),
              'n_estimators': [100], 'scale_pos_weight': np.arange(5,10)}
clf_xgbc = GridSearchCV(xgbc, param_grid = param_grid, cv = 5, scoring = 'f1',
                        n_jobs = -1)
best_clf_xgbc = clf_xgbc.fit(X_train,y_train)
clf_performance(best_clf_xgbc,'XGBClassifier')

estimators = [clf_rf, clf_gnb, clf_lda, clf_lr, clf_svc, clf_xgbc]
names = ['RandomForestClassifier',
         'gaussianNB',
         'LinearDiscriminantAnalysis',
         'LogisticRegression',
         'SVC',
         'XGBClassifier']

def test_performance(estimators,clf_names,X_test=X_test,y_test=y_test):
    
    for idx, estimator in enumerate(estimators):
        
        y_pred = estimator.predict(X_test)
        print(clf_names[idx])
        print('Test recall: {}'.format(recall_score(y_test, y_pred)))
        print('Test precision: {}'.format(precision_score(y_test, y_pred)))
        print('Test accuracy: {}\n'.format(accuracy_score(y_test, y_pred)))
        
        # Create and reshape confusion matrix data.
        matrix = confusion_matrix(y_test, y_pred)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        # Plot confusion matrix.
        plt.figure(figsize=(16,7))
        sns.set(font_scale=1.4)
        sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                    linewidths=0.2, vmin=0, vmax=1)
        class_names = ['Malicious', 'Benign']
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, class_names, rotation=25)
        plt.yticks(tick_marks2, class_names, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix for {}'.format(clf_names[idx]))
        plt.show()
    
test_performance(estimators,names)
        

# Plot ROC.
pred_prob = clf_xgbc.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob[:,1])
fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# Calculate ROC AUC.
print('ROC AUC: {}'.format(metrics.auc(fpr, tpr)))

# Plot precision-recall curve.
precision, recall, _ = precision_recall_curve(y_test, pred_prob[:,1])
plt.plot(recall, precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

# Calculate precision-recall AUC.
print('Precision-Recall AUC: {}'.format(metrics.auc(recall, precision)))
