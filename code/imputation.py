import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler



# Load dataset.
df = pd.read_csv('../output/eda/csv/df_to_impute.csv')

# Display percent null values per column.    
for i in df.columns:
    print(i, ':', 100*df[i].isna().sum()/len(df[i]))

# Create X and y variables.
X = df.copy()
y = X.pop('Type')

# Train-test split.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    stratify=y,
                                                    random_state=1)

# Categorize numerical and categorical variables for preprocessing.
numerical = ['URL_LENGTH',
             'CONTENT_LENGTH',
             'NUMBER_SPECIAL_CHARACTERS',
             'TCP_CONVERSATION_EXCHANGE',
             'DIST_REMOTE_TCP_PORT',
             'REMOTE_IPS',
             'APP_BYTES',
             'SOURCE_APP_PACKETS',
             'REMOTE_APP_PACKETS',
             'SOURCE_APP_BYTES',
             'REMOTE_APP_BYTES',
             'APP_PACKETS',
             'DNS_QUERY_TIMES']
categorical_1 = ['WHOIS_COUNTRY', 'WHOIS_STATEPRO']
categorical_2 = []
for i in X_train.columns:
    if (i not in numerical) and (i not in categorical_1):
        categorical_2.append(i)
        

        
# Create pipelines for processing
pipeline_categorical_1 = make_pipeline(
    SimpleImputer(missing_values=np.nan, strategy='constant',
                  fill_value='UNAVAILABLE'),
    OneHotEncoder(handle_unknown='ignore')
    )
pipeline_categorical_2 = make_pipeline(
    SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
    )

pipeline_numerical = make_pipeline(
    SimpleImputer(missing_values=np.nan, strategy='median'),
    StandardScaler()
    )


       
# Function to execute preprocessing for train and test sets.
def preprocess_dataset(X, categorical_1=categorical_1, categorical_2=categorical_2,
                       numerical=numerical, train=True):
    if train == True:
             
        X = pd.concat(
            [pd.DataFrame(pipeline_categorical_1.fit_transform(X[categorical_1]).toarray()),
             pd.DataFrame(pipeline_categorical_2.fit_transform(X[categorical_2]).toarray()),
             pd.DataFrame(pipeline_numerical.fit_transform(X[numerical]))
             ], axis=1)
        
        X = X.fillna(0)
        
    else:
        
        X = pd.concat(
            [pd.DataFrame(pipeline_categorical_1.transform(X[categorical_1]).toarray()),
             pd.DataFrame(pipeline_categorical_2.transform(X[categorical_2]).toarray()),
             pd.DataFrame(pipeline_numerical.transform(X[numerical]))
             ], axis=1)
        
        X = X.fillna(0) 
    
    return X
    
# Apply function to train and test data.
X_train = preprocess_dataset(X_train)
X_test = preprocess_dataset(X_test, train=False)
    
# Check for missing values.
X_train.isna().sum().sum()
X_test.isna().sum().sum()

# Check shape of matricies.
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Export split data.
X_train.to_csv('../output/imputation/X_train.csv', index=False)
X_test.to_csv('../output/imputation/X_test.csv', index=False)
y_train.to_csv('../output/imputation/y_train.csv', index=False)
y_test.to_csv('../output/imputation/y_test.csv', index=False)