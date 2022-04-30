import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.metrics import accuracy_score, recall_score, precision_score


df = pd.read_csv('../output/eda_and_cleaning/df_to_impute.csv')


df = df.iloc[:,0:39]
df_benign = df.loc[df['Type'] == 0,:]

X_train, X_valid_test, ignore_1, ignore_2 = train_test_split(df_benign,
                                                             np.zeros(len(df_benign)),
                                                             test_size=0.40,
                                                             random_state=42)

df_valid_test = pd.concat([X_valid_test, df.loc[df['Type'] == 1,:]])
X_valid_test = df_valid_test.copy()
y_valid_test = X_valid_test.pop('Type')

X_train = X_train.drop('Type',axis=1)

# temporary workspace
temp = X_train['URL_LENGTH']
temp
temp = 1/(np.power(temp,1/3.5))
sns.histplot(temp)
plt.show()

# Transformations based on X_train.
# Apply to X_train and X_valid_test.
def variable_transformation(X):
    
    df_transformed = pd.DataFrame()
    
    # URL_LENGTH transformation
    temp = X['URL_LENGTH']
    temp = 1/(np.power(temp,1/3.5))
    df_transformed['URL_LENGTH'] = temp
    
    return df_transformed

X_train = variable_transformation(X_train)
X_valid_test = variable_transformation(X_valid_test)

X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test,
                                                             test_size=0.50,
                                                             stratify=y_valid_test,
                                                             random_state=42)

def calculate_p(testing_data, X_train = X_train):
    
    """
    Calculates the probability of observing a row of data based on data that
    is known to contain benign URLS. Assumes each predictor is independent.
    
    testing_data is the X dataframe to determine p-values.
    X_train is the data with known benign websites.
    """
    
    benign_mean = X_train.mean().values
    benign_var = X_train.var().values
    
    p_values = []
    for m in range(len(testing_data)):
        p = 1
        for idx, val in enumerate(benign_mean):
            p *= norm(val, benign_var[idx]).pdf(testing_data.iloc[m,idx])
            if val == benign_mean[-1]:
                p_values.append(p)
    
    return p_values

p_values = calculate_p(X_valid)

def p_value_threshold_search(search_range, p_values, testing_targets):
    
    p_thresholds = []
    accuracy=[]
    recall = []
    precision = []    
      
    for idx_1, val_1 in enumerate(search_range):
        y_pred = []  
        for idx_2, val_2 in enumerate(p_values):
            if val_2 >= val_1:
                y_pred.append(0)
            else:
                y_pred.append(1)
        
        current_accuracy = accuracy_score(testing_targets, y_pred)
        current_recall = recall_score(testing_targets, y_pred)
        current_precision = precision_score(testing_targets, y_pred)
        
        p_thresholds.append(val_1)
        accuracy.append(current_accuracy)
        recall.append(current_recall)
        precision.append(current_precision)
            
    return p_thresholds, accuracy, recall, precision

p_thresholds,accuracy,recall,precision = p_value_threshold_search(
    np.linspace(1e-300, 1, 100), p_values, y_valid)