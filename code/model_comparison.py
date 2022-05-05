import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, roc_curve
sns.set_theme(style="whitegrid")

# Plot parameters
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True

df_no_sampling = pd.read_csv('../output/modeling/no_sampling/csv/df_scores.csv')
df_upsampling = pd.read_csv('../output/modeling/upsampling/csv/df_scores.csv')
df_downsampling = pd.read_csv('../output/modeling/downsampling/csv/df_scores.csv')

df = pd.concat([df_no_sampling, df_upsampling, df_downsampling],
               axis=0).reset_index().drop('index', axis=1)

# Plot side-by-side barplots to compare models, metrics, and sampling methods.
for idx, val in enumerate(df.columns[2:]):
    
    df = df.sort_values(by=['Model', 'Sampling',val], ascending=True)
    
    plt.figure(figsize=(10, 10))
    ax = sns.catplot(data = df, y = 'Model', x = val, hue = 'Sampling', kind='bar', legend=False)
    ax.set(xlim=(0,1))
    ax.set(ylabel=None)
    plt.gca().legend().set_title('')
    plt.legend(bbox_to_anchor=(1.025, .55), loc=2, borderaxespad=0.)
    plt.savefig('../output/modeling/model_comparison/catplot_model_{}.jpg'.format(val), bbox_inches='tight')
    plt.show()

# Load LogisticRegression and SVC models with best parameters.    
svc = pickle.load(open('../output/modeling/upsampling/models/svc_model.pkl', 'rb'))
lr = pickle.load(open('../output/modeling/upsampling/models/logisticregression_model.pkl', 'rb'))
X_test = pd.read_csv('../output/imputation/X_test.csv')
y_test = pd.read_csv('../output/imputation/y_test.csv')


# Plot ROC.
pred_prob_lr = lr.predict_proba(X_test) 
pred_prob_svc = svc.predict_proba(X_test) 
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, pred_prob_lr[:,1])
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, pred_prob_svc[:,1])
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(fpr_lr, tpr_lr, label='LogisticRegression')
ax.plot(fpr_svc, tpr_svc, label='SVC')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
ax.legend()
ax.grid(False)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.savefig('../output/modeling/model_comparison/ROC_logisticregression_svc.jpg', bbox_inches='tight')
plt.show()




