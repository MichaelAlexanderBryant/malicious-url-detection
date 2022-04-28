import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from numpy import mean, std
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn import metrics

# Load split data.
X_train = pd.read_csv('../output/imputation/X_train.csv')
y_train = pd.read_csv('../output/imputation/y_train.csv')
y_train = np.ravel(np.array(y_train))
X_test = pd.read_csv('../output/imputation/X_test.csv')
y_test = pd.read_csv('../output/imputation/y_test.csv')

# # https://wellsr.com/python/upsampling-and-downsampling-imbalanced-data-in-python/
# df['Type'].value_counts()
# df.groupby('Type').size().plot(kind='pie',
#                                        y = "Type",
#                                        label = "Malicious URL?",
#                                        autopct='%1.1f%%')

# benign_url = df[df["Type"] == 0]
# malicious_url  = df[df["Type"] == 1]

# malicious_url_upsample = resample(malicious_url,
#                                  replace=True,
#                                  n_samples=len(benign_url),
#                                  random_state=1)

# data_upsampled = pd.concat([benign_url, malicious_url_upsample])

# data_upsampled['Type'].value_counts()
# data_upsampled.groupby('Type').size().plot(kind='pie',
#                                        y = "Type",
#                                        label = "Malicious URL?",
#                                        autopct='%1.1f%%')

# Get baseline for several models.

lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5, scoring='recall')
print(mean(cv), '+/-', std(cv))

#random forest classifier with five-fold cross validation
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5, scoring = 'recall')
print(mean(cv), '+/-', std(cv))

svc = SVC(probability = True)
cv = cross_val_score(svc,X_train,y_train,cv=5, scoring='recall')
print(mean(cv), '+/-', std(cv))

gnb = GaussianNB()
cv = cross_val_score(gnb,X_train,y_train,cv=5, scoring = 'recall')
print(mean(cv), '+/-', std(cv))

bnb = BernoulliNB()
cv = cross_val_score(bnb,X_train,y_train,cv=5, scoring = 'recall')
print(mean(cv), '+/-', std(cv))


# Performance reporting function.
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: {} +/- {}'.format(str(classifier.best_score_),str(classifier.cv_results_['std_test_score'][classifier.best_index_])))
    print('Best Parameters: ' + str(classifier.best_params_))
    
# Hyperparameter tune best baseline model.
gnb = GaussianNB()
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
clf_gnb = GridSearchCV(gnb, param_grid = param_grid, cv = 5, scoring = 'recall', n_jobs = -1)
best_clf_gnb = clf_gnb.fit(X_train,y_train)
clf_performance(best_clf_gnb,'GaussianNB')

# Evaluate model with best parameters.
gnb = GaussianNB(var_smoothing = 0.2848035868435802)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print('Test recall: {}'.format(recall_score(y_test, y_pred)))
print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))


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
plt.title('Confusion Matrix for GaussianNB')
plt.show()


# Plot ROC.
pred_prob = gnb.predict_proba(X_test)
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

# Calculate AUC (ROC).
print('AUC: {}'.format(metrics.auc(fpr, tpr)))