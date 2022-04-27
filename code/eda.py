import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

# Load dataset.
df = pd.read_csv('../dataset.csv')

# Display null values and datatypes
df.info()

# Drop id column.
df = df.drop('URL', axis=1)

# Categorize variables into numerical, categorical, and date lists.
numerical = ['URL_LENGTH',
             'NUMBER_SPECIAL_CHARACTERS',
             'TCP_CONVERSATION_EXCHANGE',
             'DIST_REMOTE_TCP_PORT',
             'REMOTE_IPS',
             'APP_BYTES',
             'SOURCE_APP_PACKETS',
             'REMOTE_APP_PACKETS',
             'APP_PACKETS',
             'DNS_QUERY_TIMES']
categorical = ['CHARSET',
               'SERVER',
               'WHOIS_COUNTRY',
               'WHOIS_STATEPRO',
               'Type']
date = ['WHOIS_REGDATE',
        'WHOIS_UPDATED_DATE']

# Display numerical variable distributions.
for i in numerical:
    plt.hist(df[i])
    plt.xlabel(i)
    plt.ylabel('FREQUENCY')
    plt.show()

# Display categorical variable distributions.
for i in categorical:
    plt.bar(df[i].value_counts().index,df[i].value_counts().values)
    plt.xlabel(i)
    plt.ylabel('FREQUENCY')
    plt.show()

# Display date column formats.    
for i in date:
    print(df[i])    

# Set 'None', '0', and 'b' in date columns to NaN.
for i in date:
    df.loc[df[i]=='None', i] = np.nan
    df.loc[df[i]=='b', i] = np.nan
    df.loc[df[i]=='0', i] = np.nan
    df[i] = df[i].astype(str)

# Split date from column with date and time using regular expressions.
WHOIS_REGDATE_date = []
for idx, val in enumerate(df['WHOIS_REGDATE']):
    j = k = m = 0 
    try:
        match = re.search(r"\d+/\d+/\d+",val)
        WHOIS_REGDATE_date.append(match.group(0))
        j = 1
    except:
        pass
    try:
        match = re.search(r"\d+-\d+-\d+",val)
        temp0 = match.group(0)
        temp1 = temp0[-2:]+'/'+temp0[-5:-3]+'/'+temp0[:4]
        WHOIS_REGDATE_date.append(temp1)
        k = 1
    except:
        pass
    if val == 'nan':
        WHOIS_REGDATE_date.append(i)
        m = 1
    if (j+k+m) != 1:
        print(val)

# Check to see if all rows are accounted for
len(WHOIS_REGDATE_date) == len(df)

# Split date from column with date and time using regular expressions.
WHOIS_UPDATED_DATE_date = []
for idx, val in enumerate(df['WHOIS_UPDATED_DATE']):
    j = k = m = 0 
    try:
        match = re.search(r"\d+/\d+/\d+",val)
        WHOIS_UPDATED_DATE_date.append(match.group(0))
        j = 1
    except:
        pass
    try:
        match = re.search(r"\d+-\d+-\d+",val)
        temp0 = match.group(0)
        temp1 = temp0[-2:]+'/'+temp0[-5:-3]+'/'+temp0[:4]
        WHOIS_UPDATED_DATE_date.append(temp1)
        k = 1
    except:
        pass
    if val == 'nan':
        WHOIS_UPDATED_DATE_date.append(i)
        m = 1
    if (j+k+m) != 1:
        print(val)
  
# Check to see if all rows are accounted for
len(WHOIS_UPDATED_DATE_date) == len(df)

# Split time from column with date and time using regular expressions.
WHOIS_REGDATE_time = []
for idx, val in enumerate(df['WHOIS_REGDATE']):
    j = m = 0 
    try:
        match = re.search(r"\d+:\d+",val)
        WHOIS_REGDATE_time.append(match.group(0))
        j = 1
    except:
        pass
    if val == 'nan':
        WHOIS_REGDATE_time.append(i)
        m = 1
    if (j+m) != 1:
        print(val)

# Check to see if all rows are accounted for
len(WHOIS_REGDATE_time) == len(df)
    
 # Split time from column with date and time using regular expressions.
WHOIS_UPDATED_DATE_time = []
for idx, val in enumerate(df['WHOIS_UPDATED_DATE']):
    j = m = 0 
    try:
        match = re.search(r"\d+:\d+",val)
        WHOIS_UPDATED_DATE_time.append(match.group(0))
        j = 1
    except:
        pass
    if val == 'nan':
        WHOIS_UPDATED_DATE_time.append(i)
        m = 1
    if (j+k+m) != 1:
        print(val)
   
# Check to see if all rows are accounted for
len(WHOIS_UPDATED_DATE_time) == len(df)   


# Put parsed lists into dataframe and fix nan values.
df['REGISTER_DATE'] = WHOIS_REGDATE_date
df.loc[df['REGISTER_DATE']==139, 'REGISTER_DATE'] = np.nan
df['REGISTER_TIME'] = WHOIS_REGDATE_time
df.loc[df['REGISTER_TIME']==139, 'REGISTER_TIME'] = np.nan
df['LAST_UPDATE_DATE'] = WHOIS_UPDATED_DATE_date
df.loc[df['LAST_UPDATE_DATE']==139, 'LAST_UPDATE_DATE'] = np.nan
df['LAST_UPDATE_TIME'] = WHOIS_UPDATED_DATE_time
df.loc[df['LAST_UPDATE_TIME']==139, 'LAST_UPDATE_TIME'] = np.nan

# Convert date and time columns to datetime.
for i in ['REGISTER_DATE', 'LAST_UPDATE_DATE']:
    df[i] = pd.to_datetime(df[i], format = "%d/%m/%Y")
for i in ['REGISTER_TIME', 'LAST_UPDATE_TIME']:
    df[i] = pd.to_datetime(df[i], format = "%H:%M")

# Create seperate columns for month, day, year, hour, and minutes.
df['REGISTER_MONTH'] = df['REGISTER_DATE'].dt.month
df['REGISTER_DAY'] = df['REGISTER_DATE'].dt.day
df['REGISTER_YEAR'] = df['REGISTER_DATE'].dt.year
df['LAST_UPDATED_MONTH'] = df['LAST_UPDATE_DATE'].dt.month
df['LAST_UPDATED_DAY'] = df['LAST_UPDATE_DATE'].dt.day
df['LAST_UPDATED_YEAR'] = df['LAST_UPDATE_DATE'].dt.year
df['REGISTER_HOUR'] = df['REGISTER_TIME'].dt.hour
df['REGISTER_MINUTES'] = df['REGISTER_TIME'].dt.minute
df['LAST_UPDATED_HOUR'] = df['LAST_UPDATE_TIME'].dt.hour
df['LAST_UPDATED_MINUTES'] = df['LAST_UPDATE_TIME'].dt.minute

# Drop original dates columns.
for i in ['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'REGISTER_DATE',
          'REGISTER_TIME','LAST_UPDATE_DATE', 'LAST_UPDATE_TIME']:
    df = df.drop(i, axis=1)

# Display columns names and dtypes    
df.info()

# Drop extra column.
df = df.drop(139, axis=1)

# Lists of dates and times
dates = ['REGISTER_MONTH',
         'REGISTER_DAY',
         'REGISTER_YEAR',
         'LAST_UPDATED_MONTH',
         'LAST_UPDATED_DAY',
         'LAST_UPDATED_YEAR']
times = ['REGISTER_HOUR',
         'REGISTER_MINUTES',
         'LAST_UPDATED_HOUR',
         'LAST_UPDATED_MINUTES']

# Distribution of dates.
for i in dates:
    plt.hist(df[i])
    plt.xlabel(i)
    plt.ylabel('FREQUENCY')
    plt.show()  

# Distribution of times.    
for i in times:
    plt.hist(df[i])
    plt.xlabel(i)
    plt.ylabel('FREQUENCY')
    plt.show()

# for tomorrow:
# just cleaned dates, but is there valuable information in errors in the data?
# below for instance, states that are entered as bc and British Columbia
# should these be cleaned or are they different due to attacker? same with dates.    
for i in df['WHOIS_STATEPRO'].unique():
    print(i)