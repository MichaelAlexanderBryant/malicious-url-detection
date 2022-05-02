"""
Modeling with imbalanced classes, here is what I tried:
    1. Changed prior probabilities for naive Bayes and linear discriminant analysis
    2. Undersampled with random forest
    3. Model tuned with scoring set to recall
    4. Unequal case weights with logistic regression and xgboost
    5. Cost sensitive training with SVC
    
Randomly guessing malicious website (type=1) produces a recall of 1, but accuracy
and precision scores of 0.12128.

Recall is more important than precision, but I also wanted a better balance
than the result from scoring by recall (which is largely due to the imbalanced
dataset). I therefore chose F1 as the metric for scoring and chose the best
model based on test set recall, precision, and accuracy with recall having
highest priority.
"""

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
from sklearn.model_selection import cross_val_score
from numpy import mean, std
from sklearn.model_selection import GridSearchCV 
import pickle
from sklearn.metrics import (recall_score, accuracy_score, f1_score,
                             precision_score, confusion_matrix,
                             precision_recall_curve)
from sklearn import metrics

# Load split data.
X_train = pd.read_csv('../output/imputation/X_train.csv')
y_train = pd.read_csv('../output/imputation/y_train.csv')
y_train = np.ravel(np.array(y_train))
X_test = pd.read_csv('../output/imputation/X_test.csv')
y_test = pd.read_csv('../output/imputation/y_test.csv')

# LogisticRegression baseline.
lr = LogisticRegression(max_iter = 15000)
cv = cross_val_score(lr,X_train,y_train,cv=5, scoring='f1')
print("LogisticRegression: {} +/- {}".format(mean(cv),std(cv)))

# RandomForestClassifier.
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5, scoring = 'f1')
print("RandomForestClassifier: {} +/- {}".format(mean(cv),std(cv)))

# SVC baseline.
svc = SVC(probability = True)
cv = cross_val_score(svc,X_train,y_train,cv=5, scoring='f1')
print("SVC: {} +/- {}".format(mean(cv),std(cv)))

# GaussianNB baseline.
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train,y_train,cv=5, scoring = 'f1')
print("GaussianNB: {} +/- {}".format(mean(cv),std(cv)))

#BernoulliNB baseline.
bnb = BernoulliNB()
cv = cross_val_score(bnb,X_train,y_train,cv=5, scoring = 'f1')
print("BernoulliNB: {} +/- {}".format(mean(cv),std(cv)))

# LinearDiscriminantAnalysis baseline.
lda = LinearDiscriminantAnalysis(priors=[0.6,0.4])
cv = cross_val_score(lda,X_train,y_train,cv=5, scoring = 'f1')
print("LinearDiscriminatAnalysis: {} +/- {}".format(mean(cv),std(cv)))

# XGBClassifier baseline.
xgbc = XGBClassifier(scale_pos_weight=8,use_label_encoder = False, eval_metric='error')
cv = cross_val_score(xgbc,X_train,y_train,cv=5, scoring = 'f1')
print("XGBClassifier: {} +/- {}".format(mean(cv),std(cv)))

# Performance reporting function.
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: {} +/- {}'.format(str(classifier.best_score_),str(
        classifier.cv_results_['std_test_score'][classifier.best_index_])))
    print('Best Parameters: ' + str(classifier.best_params_))

# GridSearchCV for LogisticRegression.
# class_weight penalizes the misclasification of the minority class.
lr = LogisticRegression(max_iter = 15000)
param_grid = {
              'C' : [.75,1,1.25],
              'class_weight': [{0: 1, 1: w} for w in np.arange(1,10)]
             }
clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, scoring='f1',
                      n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train,y_train)
clf_performance(best_clf_lr,'Logistic Regression')
outfile = open('../output/modeling/no_sampling/models/logisticregression_model.pkl', 'wb')
pickle.dump(best_clf_lr,outfile)
outfile.close()

# GridSearchCV for RandomForestClassifier.    
# class_weight is a form of downsampling to accomodate class imbalance.
rf = RandomForestClassifier(random_state = 1)
param_grid =  {
                'n_estimators': np.arange(1,10,2), 
                'bootstrap': [True,False], #bagging (T) vs. pasting (F)
                'max_depth': np.arange(1,20,2),
                'max_features': ['auto','sqrt'],
                'min_samples_leaf': [2],
                'min_samples_split': np.arange(2,20,2),
                'class_weight': [{0: 1, 1: w} for w in np.arange(4,20,2)]
              }
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, scoring='f1',
                          n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,y_train)
clf_performance(best_clf_rf,'RandomForestClassifier')
outfile = open('../output/modeling/no_sampling/models/randomforestclassifier_model.pkl', 'wb')
pickle.dump(best_clf_rf,outfile)
outfile.close()

# GridSearchCV for SVC.
# class_weight is a form of cost-sensitive training to accomodate class imbalance.
svc = SVC(probability = True, random_state = 1)
param_grid = {
              'kernel': ['linear', 'poly', 'sigmoid','rbf'],
              'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
              'C': [1,10,30,50],
              'class_weight': [{0: 1, 1: w} for w in np.arange(1,10)]
             }
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, scoring = 'f1',
                       n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train,y_train)
clf_performance(best_clf_svc,'SVC')
outfile = open('../output/modeling/no_sampling/models/svc_model.pkl', 'wb')
pickle.dump(best_clf_svc,outfile)
outfile.close()

# GridSearchCV for GaussianNB.
# Adjusting prior probabilities is a way to accomodate class imbalance.
gnb = GaussianNB()
param_grid = {
              'var_smoothing': np.logspace(0,-9, num=100),
              'priors': [[1-x,x] for x in np.arange(.1,.9,.1)]
              }
clf_gnb = GridSearchCV(gnb, param_grid = param_grid, cv = 5, scoring = 'f1',
                       n_jobs = -1)
best_clf_gnb = clf_gnb.fit(X_train,y_train)
clf_performance(best_clf_gnb,'GaussianNB')
outfile = open('../output/modeling/no_sampling/models/gaussiannb_model.pkl', 'wb')
pickle.dump(best_clf_gnb,outfile)
outfile.close()

# GridSearchCV for LinearDiscriminantAnalysis.
# Adjusting prior probabilities is a way to accomodate class imbalance.
lda = LinearDiscriminantAnalysis()
param_grid = {
              'solver': ['svd'],
              'priors': [[1-x,x] for x in np.arange(.1,.9,.1)]
             }
clf_lda = GridSearchCV(lda, param_grid = param_grid, cv = 5, scoring = 'f1',
                       n_jobs = -1)
best_clf_lda = clf_lda.fit(X_train,y_train)
clf_performance(best_clf_lda,'LinearDiscriminantAnalysis')
outfile = open('../output/modeling/no_sampling/models/lineardiscriminantanalysis_model.pkl', 'wb')
pickle.dump(best_clf_lda,outfile)
outfile.close()

# GridSearchCV for XGBClassifier.
# Adjusting scale_pos_weight gives more weight to the positive/minority class.
xgbc = XGBClassifier(eval_metric='error')
param_grid = {
              'learning_rate': [0.001],
              'max_depth': np.arange(1,20,2),
              'n_estimators': [300],
              'scale_pos_weight': np.arange(1,7,1)
              }
clf_xgbc = GridSearchCV(xgbc, param_grid = param_grid, cv = 5, scoring = 'f1',
                        n_jobs = -1)
best_clf_xgbc = clf_xgbc.fit(X_train,y_train)
clf_performance(best_clf_xgbc,'XGBClassifier')
outfile = open('../output/modeling/no_sampling/models/xgbclassifier_model.pkl', 'wb')
pickle.dump(best_clf_xgbc,outfile)
outfile.close()

# Lists of estimators and names.
estimators = [best_clf_rf, best_clf_gnb, best_clf_lda, best_clf_lr, best_clf_svc, best_clf_xgbc]
names = ['RandomForestClassifier',
         'GaussianNB',
         'LinearDiscriminantAnalysis',
         'LogisticRegression',
         'SVC',
         'XGBClassifier']

# Function to compute metrics, ROC, and confusion matricies.
def test_performance(estimators,clf_names,X_test=X_test,y_test=y_test):
    
    df_scores = pd.DataFrame()
    
    for idx, estimator in enumerate(estimators):
        
        y_pred = estimator.predict(X_test)
        print(clf_names[idx])
        test_recall = recall_score(y_test, y_pred)
        print('Test recall: {}'.format(test_recall))
        test_precision = precision_score(y_test, y_pred)
        print('Test precision: {}'.format(test_precision))
        test_f1 = f1_score(y_test, y_pred)
        print('Test F1: {}'.format(test_f1))
        test_accuracy = accuracy_score(y_test, y_pred)
        print('Test accuracy: {}'.format(test_accuracy))
        
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
        plt.savefig('../output/modeling/no_sampling/images/confusion_matrix_{}.jpg'.format(clf_names[idx]), bbox_inches='tight')
        plt.show()
        
        # Plot ROC.
        pred_prob = estimator.predict_proba(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob[:,1])
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC for {}'.format(clf_names[idx]))
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.savefig('../output/modeling/no_sampling/images/ROC_{}.jpg'.format(clf_names[idx]), bbox_inches='tight')
        plt.show()

        # Calculate ROC AUC.
        ROC_AUC = metrics.auc(fpr, tpr)
        print('ROC AUC for {}: {}'.format(clf_names[idx],ROC_AUC))

        # Plot precision-recall curve.
        precision, recall, _ = precision_recall_curve(y_test, pred_prob[:,1])
        plt.plot(recall, precision, marker='.', label='{}'.format(clf_names[idx]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig('../output/modeling/no_sampling/images/PR_curve_{}.jpg'.format(clf_names[idx]), bbox_inches='tight')
        plt.show()

        # Calculate precision-recall AUC.
        PR_AUC = metrics.auc(recall, precision)
        print('Precision-Recall AUC for {}: {}\n'.format(clf_names[idx], PR_AUC))
        
        # Create a list of all scores.
        clf_scores = ['No sampling', clf_names[idx], test_recall,
                      test_precision, test_f1, test_accuracy, ROC_AUC, PR_AUC]
        
        # Create a dataframe with scores for later analysis.
        if len(df_scores) == 0:
            
            df_scores = pd.DataFrame([clf_scores], columns = ['Sampling',
                                                            'Model',
                                                            'Recall',
                                                            'Precision',
                                                            'F1',
                                                            'Accuracy',
                                                            'ROC AUC',
                                                            'Precision-Recall AUC'])
        
        else: 
            df_scores.loc[len(df_scores)] = clf_scores
        
    return df_scores

# Compute metrics with test set for all models with best parameters.  
df_scores = test_performance(estimators,names)

# Output csv file for all models and metrics.
df_scores.to_csv('../output/modeling/no_sampling/csv/df_scores.csv', index=False)