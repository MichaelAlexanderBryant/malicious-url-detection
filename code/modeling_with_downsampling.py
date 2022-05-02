"""
Modeling with downsampling.
    
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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import itertools as it
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from xgboost import XGBClassifier
from numpy import mean, std
import pickle
from sklearn.metrics import (recall_score, accuracy_score, f1_score,
                             precision_score, confusion_matrix,
                             precision_recall_curve)
from sklearn import metrics

# Load split data.
X_train_valid = pd.read_csv('../output/imputation/X_train.csv')
y_train_valid = pd.read_csv('../output/imputation/y_train.csv')
X_test = pd.read_csv('../output/imputation/X_test.csv')
y_test = pd.read_csv('../output/imputation/y_test.csv')

# Function that computes cross-validation scores by first splitting the data
# into training and validation data and then downsamples the training data.
def downsampleCV(clf, X_train_valid, y_train_valid, cv, scoring):

    cv_scores = []
    
    skf = StratifiedKFold(n_splits=cv, random_state=None, shuffle=False)
    for train_index, valid_index in skf.split(X_train_valid, y_train_valid):

        X_train, X_valid = X_train_valid.iloc[train_index,:], X_train_valid.iloc[valid_index,:]
        y_train, y_valid = y_train_valid.iloc[train_index,:], y_train_valid.iloc[valid_index,:]
        
        train_data = pd.concat([X_train,y_train],axis=1)

        malicious_websites = train_data[train_data["Type"] == 1]
        benign_websites = train_data[train_data["Type"] == 0]

        benign_downsample = resample(benign_websites,
                     replace=True,
                     n_samples=len(malicious_websites),
                     random_state=1)

        train_data = pd.concat([benign_downsample, malicious_websites], axis=0)

        X_train = train_data.copy()
        y_train = X_train.pop('Type')
        
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_valid)
        
        if scoring == 'f1':
            cv_scores.append(f1_score(y_valid, y_pred))
            
    return cv_scores
    


# LogisticRegression baseline.
lr = LogisticRegression(max_iter = 15000)
cv = downsampleCV(lr,X_train_valid,y_train_valid,cv=5, scoring='f1')
print("LogisticRegression: {} +/- {}".format(mean(cv),std(cv)))

# RandomForestClassifier baseline.
rf = RandomForestClassifier(random_state = 1)
cv = downsampleCV(rf,X_train_valid,y_train_valid,cv=5, scoring='f1')
print("RandomForestClassifier: {} +/- {}".format(mean(cv),std(cv)))

# SVC baseline.
svc = SVC(probability = True)
cv = downsampleCV(svc,X_train_valid,y_train_valid,cv=5, scoring='f1')
print("SVC: {} +/- {}".format(mean(cv),std(cv)))

# GaussianNB baseline.
gnb = GaussianNB()
cv = downsampleCV(gnb,X_train_valid,y_train_valid,cv=5, scoring='f1')
print("GaussianNB: {} +/- {}".format(mean(cv),std(cv)))

# BernoulliNB baseline.
bnb = BernoulliNB()
cv = downsampleCV(bnb,X_train_valid,y_train_valid,cv=5, scoring='f1')
print("BernoulliNB: {} +/- {}".format(mean(cv),std(cv)))

# LinearDiscriminantAnalysis baseline.
lda = LinearDiscriminantAnalysis()
cv = downsampleCV(lda,X_train_valid,y_train_valid,cv=5, scoring='f1')
print("LinearDiscriminatAnalysis: {} +/- {}".format(mean(cv),std(cv)))

# XGBClassifier baseline.
xgbc = XGBClassifier(use_label_encoder = False, eval_metric='error')
cv = downsampleCV(xgbc,X_train_valid,y_train_valid,cv=5, scoring='f1')
print("XGBClassifier: {} +/- {}".format(mean(cv),std(cv)))

# Function that does a gridsearch (similar to GridSearchCV), but downsamples
# the majority class training set after splitting the validation set from it.
def downsample_gridsearchCV(clf, X, y, cv, scoring, param_grid):
    
    parameters = list(param_grid.keys())
    combinations = list(it.product(*(param_grid[Name] for Name in param_grid)))
    
    parameter_combinations = []
    for idx, val in enumerate(combinations):
        parameter_set = dict(zip(parameters,val))
        parameter_combinations.append(parameter_set)
        

    df_scores=pd.DataFrame()
    skf = StratifiedKFold(n_splits=cv, random_state=None, shuffle=False)
    for train_index, valid_index in skf.split(X_train_valid, y_train_valid):

        X_train, X_valid = X_train_valid.iloc[train_index,:], X_train_valid.iloc[valid_index,:]
        y_train, y_valid = y_train_valid.iloc[train_index,:], y_train_valid.iloc[valid_index,:]
        
        train_data = pd.concat([X_train,y_train],axis=1)

        malicious_websites = train_data[train_data["Type"] == 1]
        benign_websites = train_data[train_data["Type"] == 0]

        benign_downsample = resample(benign_websites,
                     replace=True,
                     n_samples=len(malicious_websites),
                     random_state=1)

        train_data = pd.concat([benign_downsample, malicious_websites], axis=0)

        X_train = train_data.copy()
        y_train = X_train.pop('Type')
    
        scores = []
        for i in parameter_combinations:
            
            clf.set_params(**i)
            y_pred = clf.fit(X_train, y_train).predict(X_valid)
            
            if scoring == 'f1':
                scores.append(f1_score(y_valid,y_pred))
                
        if len(df_scores) == 0:
            df_scores = pd.DataFrame([scores])
        else:
            df_scores = df_scores.append(pd.DataFrame([scores]),ignore_index=True)
            
    best_parameters_idx = np.argmax(np.mean(df_scores,axis=0))
    best_parameters = parameter_combinations[best_parameters_idx]
    mean_score = np.mean(df_scores,axis=0)[best_parameters_idx]
    std_score = np.std(df_scores,axis=0)[best_parameters_idx]
            
    return best_parameters, mean_score, std_score
            
# Performance reporting function.
def clf_performance(best_parameters, mean_score, std_score, model_name):
    print(model_name)
    print('Best Score: {} +/- {}'.format(str(mean_score),str(std_score)))
    print('Best Parameters: ' + str(best_parameters))

# Function for fitting all training data with optimal parameters found with
# downsample_gridsearchCV.
def fit_all_training_data(clf, best_parameters, X_train_valid=X_train_valid,
                          y_train_valid=y_train_valid):

    best_clf = clf.set_params(**best_parameters).fit(X_train_valid, y_train_valid.values.ravel())
    
    return best_clf
    
# downsample_gridsearchCV for LogisticRegression.  
lr = LogisticRegression(max_iter=15000)
param_grid = {
              'C' : np.arange(0.1,1,0.1),
              'class_weight': [{0: 1, 1: w} for w in np.arange(1,10)]
             }
best_parameters, mean_score, std_score = downsample_gridsearchCV(
    lr,X_train_valid, y_train_valid, cv=5,
    scoring = 'f1', param_grid=param_grid)
clf_performance(best_parameters, mean_score, std_score,'\nLogistic Regression')
clf_lr = fit_all_training_data(lr,best_parameters)
outfile = open('../output/modeling/downsampling/models/logisticregression_model.pkl', 'wb')
pickle.dump(clf_lr,outfile)
outfile.close()    

# downsample_gridsearchCV for RandomForestClassifier. 
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
best_parameters, mean_score, std_score = downsample_gridsearchCV(
    rf,X_train_valid, y_train_valid, cv=5,
    scoring = 'f1', param_grid=param_grid)
clf_performance(best_parameters, mean_score, std_score,'\nRandomForestClassifier')
clf_rf = fit_all_training_data(rf,best_parameters)
outfile = open('../output/modeling/downsampling/models/randomforestclassifier_model.pkl', 'wb')
pickle.dump(clf_rf,outfile)
outfile.close()  

# downsample_gridsearchCV for SVC.
svc = SVC(probability = True, random_state = 1)
param_grid = {
              'kernel': ['linear', 'poly', 'sigmoid','rbf'],
              'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
              'C': np.arange(.5,1.5,.1),
              'class_weight': [{0: 1, 1: w} for w in np.arange(1,10)]
             }
best_parameters, mean_score, std_score = downsample_gridsearchCV(
    svc,X_train_valid, y_train_valid, cv=5,
    scoring = 'f1', param_grid=param_grid)
clf_performance(best_parameters, mean_score, std_score,'SVC')
clf_svc = fit_all_training_data(svc,best_parameters)
outfile = open('../output/modeling/downsampling/models/svc_model.pkl', 'wb')
pickle.dump(clf_svc,outfile)
outfile.close()  

# downsample_gridsearchCV for GaussianNB.
gnb = GaussianNB()
param_grid = {
              'var_smoothing': np.logspace(0,-9, num=100),
              'priors': [[1-x,x] for x in np.arange(.1,.9,.1)]
              }
best_parameters, mean_score, std_score = downsample_gridsearchCV(
    gnb,X_train_valid, y_train_valid, cv=5,
    scoring = 'f1', param_grid=param_grid)
clf_performance(best_parameters, mean_score, std_score,'\nGaussianNB')
clf_gnb = fit_all_training_data(gnb,best_parameters)
outfile = open('../output/modeling/downsampling/models/gaussiannb_model.pkl', 'wb')
pickle.dump(clf_gnb,outfile)
outfile.close()  

# downsample_gridsearchCV for LinearDiscriminantAnalysis.
lda = LinearDiscriminantAnalysis()
param_grid = {
              'solver': ['svd'],
              'priors': [[1-x,x] for x in np.arange(.1,.9,.1)]
             }
best_parameters, mean_score, std_score = downsample_gridsearchCV(
    lda,X_train_valid, y_train_valid, cv=5,
    scoring = 'f1', param_grid=param_grid)
clf_performance(best_parameters, mean_score, std_score,'\nLinearDiscriminantAnalysis')
clf_lda = fit_all_training_data(lda,best_parameters)
outfile = open('../output/modeling/downsampling/models/lineardiscriminant_model.pkl', 'wb')
pickle.dump(clf_lda,outfile)
outfile.close() 

# downsample_gridsearchCV for XGBClassifier.
xgbc = XGBClassifier(eval_metric='error', verbosity=0)
param_grid = {
              'learning_rate': [0.001],
              'max_depth': np.arange(1,20,2),
              'n_estimators': [300],
              'scale_pos_weight': np.arange(1,7,1)
              }
best_parameters, mean_score, std_score = downsample_gridsearchCV(
    xgbc,X_train_valid, y_train_valid, cv=5,
    scoring = 'f1', param_grid=param_grid)
clf_performance(best_parameters, mean_score, std_score,'\nXGBClassifier')
clf_xgbc = fit_all_training_data(xgbc,best_parameters)
outfile = open('../output/modeling/downsampling/models/xgbclassifier_model.pkl', 'wb')
pickle.dump(clf_xgbc,outfile)
outfile.close() 

# Lists of estimators and names.
estimators = [clf_rf, clf_gnb, clf_lda, clf_lr, clf_svc, clf_xgbc]
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
        plt.savefig('../output/modeling/downsampling/images/confusion_matrix_{}.jpg'.format(clf_names[idx]), bbox_inches='tight')
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
        plt.savefig('../output/modeling/downsampling/images/ROC_{}.jpg'.format(clf_names[idx]), bbox_inches='tight')
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
        plt.savefig('../output/modeling/downsampling/images/PR_curve_{}.jpg'.format(clf_names[idx]), bbox_inches='tight')
        plt.show()

        # Calculate precision-recall AUC.
        PR_AUC = metrics.auc(recall, precision)
        print('Precision-Recall AUC for {}: {}\n'.format(clf_names[idx], PR_AUC))
        
        # Create a list of all scores.
        clf_scores = ['Downsampling', clf_names[idx], test_recall,
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
df_scores.to_csv('../output/modeling/downsampling/csv/df_scores.csv', index=False)