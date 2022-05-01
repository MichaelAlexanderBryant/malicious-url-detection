"""
This is an anomaly detection system that:
    1. takes ~60% of the benign website data (which is the training data)
    2. imputes the data
    3. transforms that data into Gaussian distributions
    4. takes the mean and variance of each variable from the training data
    5. combines the remaining ~40% of benign website data and all malicious
        website data (which is the validation and test data)
    6. performs the same transformations that were done on the training data
    7. uses a Gaussian distribution for each variable from the training set
        (with the calculated means and variances from step 3) to obtain the probability
        of seeing a variable with that magnitude if it were from the benign distribution
    8. multiplies the probabilities together to find the probability of seeing such
        a row (assumes independence of variables)
    9. searches p-value thresholds to find the value that has the highest metric
        (e.g., F1, recall, precision, accuarcy).
        
    I left off on making the transformations.
    Need to impliment cross-validation either using scitkit-learn or using a function
    I started at the bottom.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.metrics import accuracy_score, recall_score, precision_score


df = pd.read_csv('../output/eda_and_cleaning/df_to_impute.csv')

df = df.iloc[:,0:39]
df = df.loc[:,df.dtypes != 'object']

df_benign = df.loc[df['Type'] == 0,:]

X_train, X_valid_test, ignore_1, ignore_2 = train_test_split(df_benign,
                                                             np.zeros(len(df_benign)),
                                                             test_size=0.40,
                                                             random_state=42)

df_valid_test = pd.concat([X_valid_test, df.loc[df['Type'] == 1,:]])
X_valid_test = df_valid_test.copy()
y_valid_test = X_valid_test.pop('Type')
X_train = X_train.drop('Type',axis=1)

imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

X_train = pd.DataFrame(imp_median.fit_transform(X_train), columns = df_benign.columns[df_benign.columns != 'Type'])
X_valid_test = pd.DataFrame(imp_median.transform(X_valid_test), columns = df_benign.columns[df_benign.columns != 'Type'])

X_train.info()

X_train['TCP_CONVERSATION_EXCHANGE'].describe()

# temporary workspace
temp = X_train['TCP_CONVERSATION_EXCHANGE']
sns.histplot(np.power(temp+5,1/10), bins = 50)
plt.show()

# Transformations based on X_train.
# Apply to X_train and X_valid_test.
def variable_transformation(X):
    
    df_transformed = pd.DataFrame()
    
    # URL_LENGTH
    temp = X['URL_LENGTH']
    temp = 1/np.power(temp,1/3.5)
    df_transformed['URL_LENGTH'] = temp
    
    # NUMBER_SPECIAL_CHARACTERS
    temp = X['NUMBER_SPECIAL_CHARACTERS']
    temp = 1/np.power(temp,1.8)
    df_transformed['NUMBER_SPECIAL_CHARACTERS'] = temp
    
    # CONTENT_LENGTH
    temp = X['CONTENT_LENGTH']
    temp = np.power(temp,1/7)
    df_transformed['CONTENT_LENGTH'] = temp   
    
    # TCP_CONVERSATION_EXCHANGE
    temp = X['TCP_CONVERSATION_EXCHANGE']
    temp = 1/np.power(temp,1/5)
    df_transformed['TCP_CONVERSATION_EXCHANGE'] = temp       
    
    
    return df_transformed

X_train = variable_transformation(X_train)
X_valid_test = variable_transformation(X_valid_test)

X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test,
                                                             test_size=0.50,
                                                             stratify=y_valid_test,
                                                             random_state=42)

np.min(X_valid)
np.max(X_valid)
np.isnan(X_valid).sum()

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

X_valid

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
    np.linspace(1e-600, 1e-400, 10), p_values, y_valid)
