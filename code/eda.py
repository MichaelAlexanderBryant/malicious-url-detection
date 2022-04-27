import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

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
        WHOIS_REGDATE_date.append(val)
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
        WHOIS_UPDATED_DATE_date.append(val)
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
        WHOIS_REGDATE_time.append(val)
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
        WHOIS_UPDATED_DATE_time.append(val)
        m = 1
    if (j+k+m) != 1:
        print(val)
   
# Check to see if all rows are accounted for
len(WHOIS_UPDATED_DATE_time) == len(df)   


# Put parsed lists into dataframe and fix nan values.
df['REGISTER_DATE'] = WHOIS_REGDATE_date
df.loc[df['REGISTER_DATE']=='nan', 'REGISTER_DATE'] = np.nan
df['REGISTER_TIME'] = WHOIS_REGDATE_time
df.loc[df['REGISTER_TIME']=='nan', 'REGISTER_TIME'] = np.nan
df['LAST_UPDATE_DATE'] = WHOIS_UPDATED_DATE_date
df.loc[df['LAST_UPDATE_DATE']=='nan', 'LAST_UPDATE_DATE'] = np.nan
df['LAST_UPDATE_TIME'] = WHOIS_UPDATED_DATE_time
df.loc[df['LAST_UPDATE_TIME']=='nan', 'LAST_UPDATE_TIME'] = np.nan

# Convert date and time columns to datetime.
for i in ['REGISTER_DATE', 'LAST_UPDATE_DATE']:
    df[i] = pd.to_datetime(df[i], format = "%d/%m/%Y")
for i in ['REGISTER_TIME', 'LAST_UPDATE_TIME']:
    df[i] = pd.to_datetime(df[i], format = "%H:%M")
    
df.info()

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

# Create features based on day of year, day of month, day of week, and weekday.
df['REGISTER_DAYOFYEAR'] = df['REGISTER_DATE'].dt.dayofyear
df['REGISTER_DAYOFMONTH'] = df['REGISTER_DATE'].dt.days_in_month
df['REGISTER_DAYOFWEEK'] = df['REGISTER_DATE'].dt.dayofweek
df['REGISTER_WEEKDAY'] = df['REGISTER_DATE'].dt.weekday
df['REGISTER_WORKINGDAY'] = (df['REGISTER_DATE'].dt.weekday <= 4).astype(int)
df['REGISTER_WEEKEND'] = (df['REGISTER_DATE'].dt.weekday >= 5).astype(int)
df['LAST_UPDATED_DAYOFYEAR'] = df['LAST_UPDATE_DATE'].dt.dayofyear
df['LAST_UPDATED_DAYOFMONTH'] = df['LAST_UPDATE_DATE'].dt.days_in_month
df['LAST_UPDATED_DAYOFWEEK'] = df['LAST_UPDATE_DATE'].dt.dayofweek
df['LAST_UPDATED_WEEKDAY'] = df['LAST_UPDATE_DATE'].dt.weekday
df['LAST_UPDATED_WORKINGDAY'] = (df['LAST_UPDATE_DATE'].dt.weekday == 4).astype(int)
df['LAST_UPDATED_WEEKEND'] = (df['LAST_UPDATE_DATE'].dt.weekday >= 5).astype(int)



# Drop original dates columns.
for i in ['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'REGISTER_DATE',
          'REGISTER_TIME','LAST_UPDATE_DATE', 'LAST_UPDATE_TIME']:
    df = df.drop(i, axis=1)

# Display columns names and dtypes    
df.info()

# Lists of dates and times
dates = ['REGISTER_MONTH',
         'REGISTER_DAY',
         'REGISTER_YEAR',
         'REGISTER_DAYOFYEAR',
         'REGISTER_DAYOFMONTH',
         'REGISTER_DAYOFWEEK',
         'REGISTER_WEEKDAY',
         'REGISTER_WORKINGDAY',
         'REGISTER_WEEKEND',
         'LAST_UPDATED_MONTH',
         'LAST_UPDATED_DAY',
         'LAST_UPDATED_YEAR',
         'LAST_UPDATED_DAYOFYEAR',
         'LAST_UPDATED_DAYOFMONTH',
         'LAST_UPDATED_DAYOFWEEK',
         'LAST_UPDATED_WEEKDAY',
         'LAST_UPDATED_WORKINGDAY',
         'LAST_UPDATED_WEEKEND',]
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

<<<<<<< HEAD
# Display unique in categorical variables.
for j in categorical:
    print('-----------',j,'-----------')
    for i in df[j].unique():
        print(i)

# Set values to upper/lower cases.        
df['CHARSET'] = df['CHARSET'].str.upper()
df['SERVER'] = df['SERVER'].str.lower()
df['WHOIS_COUNTRY'] = df['WHOIS_COUNTRY'].str.upper()
df['WHOIS_STATEPRO'] = df['WHOIS_STATEPRO'].str.upper()

# Find server names without version numbers and slashes.
before_first_slash = re.compile(r"([a-zA-Z0-9_]*)[^/]*")
servers = []
for idx,val in enumerate(df['SERVER'].unique()):
    server_split = str(val).split(' ')
    for i in server_split:
        item = before_first_slash.search(i.replace('(','').replace(')',''))
        servers.append(item.group())

# Create a series of unique servers and remove accidental regex identifications.
unique_servers = pd.Series(servers).unique()
unique_servers = unique_servers[unique_servers != '']
unique_servers = unique_servers[unique_servers != 'nan']
unique_servers = unique_servers[unique_servers != 'none']
unique_servers = unique_servers[unique_servers != 'web']
unique_servers = unique_servers[unique_servers != 'server']
unique_servers = unique_servers[unique_servers != '+']
unique_servers = unique_servers[unique_servers != '5.0.30']
unique_servers = unique_servers[unique_servers != '1.12.2']
unique_servers = unique_servers[unique_servers != '2.6.8;']
unique_servers = unique_servers[unique_servers != '.v01']
unique_servers = unique_servers[unique_servers != '&']
unique_servers = unique_servers[unique_servers != '294']
unique_servers = unique_servers[unique_servers != '999']
unique_servers = unique_servers[unique_servers != 'xxxxxxxxxxxxxxxxxxxxxx']
unique_servers = unique_servers[unique_servers != 'my']
unique_servers = unique_servers[unique_servers != 'arse']

unique_servers

# Create a binary matrix of the unique servers.
binary_servers = pd.DataFrame(columns = unique_servers)
for idx,val in enumerate(df['SERVER']):
    temp = []
    for i in unique_servers:
        if i in str(val):
            temp.append(1)
        else:
            temp.append(0)
    binary_servers.loc[idx] = temp

# Display top 10 most common servers.
binary_servers.sum().sort_values(ascending=False)[0:10]

# Concatenate binary servers df with original df and drop original SERVER column
df = pd.concat([df, binary_servers], axis=1)
df = df.drop('SERVER', axis=1)

# Remove SERVER from categorical list, add new columns.
categorical.remove('SERVER')
categorical = categorical + list(binary_servers.columns)

# Display unique in categorical variables for remaining columns to clean.
for j in categorical[1:3]:
    print('-----------',j,'-----------')
    for i in df[j].unique():
        print(i)

df.loc[df['WHOIS_COUNTRY'] == 'NONE', 'WHOIS_COUNTRY'] = np.nan      
df.loc[df['WHOIS_COUNTRY'] == 'CYPRUS', 'WHOIS_COUNTRY'] = 'CY'
df.loc[df['WHOIS_COUNTRY'] == 'UNITED KINGDOM', 'WHOIS_COUNTRY'] = 'UK'        
df.loc[df['WHOIS_COUNTRY'] == "[U'GB'; U'UK']", 'WHOIS_COUNTRY'] = 'UK' 

# Display unique values in WHOIS_STATEPRO
for i in df['WHOIS_STATEPRO'].unique():
    print(i)

# Dictionary for converting US state abbreviations to full name.
us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

# Replace US State abbreviations with full name.
for key, value in us_state_to_abbrev.items():
    df.loc[df['WHOIS_STATEPRO'] == value, 'WHOIS_STATEPRO'] = key
 
# Dictionary for replacing Canadian province abbreviations with full name.
can_province_abbrev = {
  'Alberta': 'AB',
  'British Columbia': 'BC',
  'Manitoba': 'MB',
  'New Brunswick': 'NB',
  'Newfoundland and Labrador': 'NL',
  'Northwest Territories': 'NT',
  'Nova Scotia': 'NS',
  'Nunavut': 'NU',
  'Ontario': 'ON',
  'Prince Edward Island': 'PE',
  'Quebec': 'QC',
  'Saskatchewan': 'SK',
  'Yukon': 'YT'
}

# Replace Canadian province abbreviations with full name.
for key, value in can_province_abbrev.items():
    df.loc[df['WHOIS_STATEPRO'] == value, 'WHOIS_STATEPRO'] = key

# Convert state to uppercase.
df['WHOIS_STATEPRO'] = df['WHOIS_STATEPRO'].str.upper()

# Display unique values in WHOIS_STATEPRO
for i in df['WHOIS_STATEPRO'].unique():
    print(i)

# Finish cleaning states by hand.
df.loc[df['WHOIS_STATEPRO'] == 'NONE', 'WHOIS_STATEPRO'] = np.nan 
df.loc[df['WHOIS_STATEPRO'] == 'WC1N', 'WHOIS_STATEPRO'] = 'LONDON'
df.loc[df['WHOIS_STATEPRO'] == 'UK', 'WHOIS_STATEPRO'] = np.nan
df.loc[df['WHOIS_STATEPRO'] == 'P', 'WHOIS_STATEPRO'] = np.nan
df.loc[df['WHOIS_STATEPRO'] == 'QLD', 'WHOIS_STATEPRO'] = 'QUEENSLAND'
df.loc[df['WHOIS_STATEPRO'] == '--', 'WHOIS_STATEPRO'] = np.nan
df.loc[df['WHOIS_STATEPRO'] == 'NSW', 'WHOIS_STATEPRO'] = 'NEW SOUTH WALES'
df.loc[df['WHOIS_STATEPRO'] == 'VIC', 'WHOIS_STATEPRO'] = 'VICTORIA'
df.loc[df['WHOIS_STATEPRO'] == '6110021', 'WHOIS_STATEPRO'] = 'UJI'
df.loc[df['WHOIS_STATEPRO'] == 'NOT APPLICABLE', 'WHOIS_STATEPRO'] = np.nan
df.loc[df['WHOIS_STATEPRO'] == 'ILOCOS NORTE R3', 'WHOIS_STATEPRO'] = 'ILOCOS NORTE'
df.loc[df['WHOIS_STATEPRO'] == 'WIDESTEP@MAIL.RU', 'WHOIS_STATEPRO'] = np.nan
df.loc[df['WHOIS_STATEPRO'] == 'ZH', 'WHOIS_STATEPRO'] = 'ZURICH'
df.loc[df['WHOIS_STATEPRO'] == '-', 'WHOIS_STATEPRO'] = np.nan
df.loc[df['WHOIS_STATEPRO'] == 'CH', 'WHOIS_STATEPRO'] = np.nan
df.loc[df['WHOIS_STATEPRO'] == 'TOKYO-TO', 'WHOIS_STATEPRO'] = 'TOKYO'
df.loc[df['WHOIS_STATEPRO'] == 'CO. DUBLIN', 'WHOIS_STATEPRO'] = 'DUBLIN'
df.loc[df['WHOIS_STATEPRO'] == 'OTHER', 'WHOIS_STATEPRO'] = np.nan

# Display unique values in WHOIS_STATEPRO
for i in df['WHOIS_STATEPRO'].unique():
    print(i)
    
# Display percent null values per column.    
for i in df.columns:
    print(i, ':', 100*df[i].isna().sum()/len(df[i]))
