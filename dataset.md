# I did more than what was shown here, so things may not align very well but it should be able to tell you how I did it.

### #1 converted the DATE and TIME columns into a single datetime column, adjusted the time from Moscow Time to Eastern Time, and removed the weekends.



```python

import pandas as pd
from pytz import timezone

# Path to your CSV file
file_path = '/Users/ronangrant/Downloads/EURUSD_historical_data4.csv'

# Read the CSV file
df = pd.read_csv(file_path, delimiter=';')

# Combine DATE and TIME into a single datetime column and convert to datetime object
df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%m/%d/%y %H:%M:%S')

# Assuming the time is in Moscow Time (MSK)
moscow = timezone('Europe/Moscow')
eastern = timezone('US/Eastern')

# Convert the datetime to Eastern Time
df['datetime'] = df['datetime'].dt.tz_localize(moscow).dt.tz_convert(eastern)

# Remove Fridays, Saturdays, and Sundays
df['weekday'] = df['datetime'].dt.weekday
df = df[~df['weekday'].isin([4, 5, 6])]

# Drop the weekday and original DATE and TIME columns
df.drop(['weekday', '<DATE>', '<TIME>'], axis=1, inplace=True)

# Save to a new CSV file
df.to_csv('/Users/ronangrant/Downloads/EURUSD_modified.csv', index=False)

```


```python
import pandas as pd

# Path to your CSV file
file_path = '/Users/ronangrant/Vault/training_data/original/EURUSD_modified.csv'

# Read the CSV file and display the first few rows
df = pd.read_csv(file_path, delimiter=';')
print(df.head())

```

      <TICKER>,<PER>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,datetime
    0  EURUSD,1,1.04072,1.04088,1.04015,1.04052,7317,...       
    1  EURUSD,1,1.0405,1.0409,1.03965,1.03999,7020,20...       
    2  EURUSD,1,1.03998,1.04055,1.0398,1.04039,5505,2...       
    3  EURUSD,1,1.04036,1.04062,1.04015,1.0404,2764,2...       
    4  EURUSD,1,1.04049,1.04058,1.0402,1.04054,1564,2...       


### Calculates the 50-minute SMA and EMA, and removes rows with NaN values and then saves it to a new csv



```python
import pandas as pd

# Load your CSV file
df = pd.read_csv('/Users/ronangrant/Vault/training_data/original/EURUSD_modified.csv')

# Remove angle brackets from column names
df.columns = df.columns.str.strip('<>')

# Convert the 'datetime' column to a datetime object, handling timezones
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

# Calculate the Simple Moving Average (SMA) for 50 minutes
df['SMA_50'] = df['CLOSE'].rolling(window=50).mean()

# Calculate the Exponential Moving Average (EMA) for 50 minutes
df['EMA_50'] = df['CLOSE'].ewm(span=50, adjust=False).mean()

# Remove rows with NaN values
df.dropna(inplace=True)

# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_with_indicators.csv', index=False)

```

### Rounds the SMA and EMA


```python
# Round the SMA and EMA values to 5 decimal places
df['SMA_50'] = df['SMA_50'].round(5)
df['EMA_50'] = df['EMA_50'].round(5)

# Save the updated DataFrame again
df.to_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_with_rounded_indicators.csv', index=False)

```

### Calculates the RSI, MACD, and Bollinger Bands, then adds them to the dataset


```python
import pandas as pd
import numpy as np

# Load your CSV file
df = pd.read_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_with_rounded_indicators.csv')

# Function to calculate RSI
def compute_rsi(data, window=14):
    diff = data.diff(1).dropna()
    gain = 0 * diff
    loss = 0 * diff
    gain[diff > 0] = diff[diff > 0]
    loss[diff < 0] = -diff[diff < 0]
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI for 14 periods
df['RSI_14'] = compute_rsi(df['CLOSE'], window=14)

# Calculate MACD and MACD Signal
df['MACD'] = df['CLOSE'].ewm(span=12, adjust=False).mean() - df['CLOSE'].ewm(span=26, adjust=False).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Calculate Bollinger Bands
df['SMA_20'] = df['CLOSE'].rolling(window=20).mean()
df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['CLOSE'].rolling(window=20).std()
df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['CLOSE'].rolling(window=20).std()

# Remove rows with NaN values
df.dropna(inplace=True)

# Round off columns to 5 decimal places
cols_to_round = ['SMA_50', 'EMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'SMA_20', 'Bollinger_Upper', 'Bollinger_Lower']
df[cols_to_round] = df[cols_to_round].round(5)

# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_with_more_indicators.csv', index=False)

```

### This adjustment will make the MACD and MACD Signal values more readable and potentially more useful for your model.


```python
# Round off columns to 5 decimal places
cols_to_round = ['SMA_50', 'EMA_50', 'RSI_14', 'SMA_20', 'Bollinger_Upper', 'Bollinger_Lower']
df[cols_to_round] = df[cols_to_round].round(5)

# Round MACD and MACD_Signal to 6 decimal places for better precision
df['MACD'] = df['MACD'].round(6)
df['MACD_Signal'] = df['MACD_Signal'].round(6)

# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_with_rounded_indicators.csv', index=False)

```

### Dataset will have an additional column ('Target') that indicates whether the price will go up (1) or down (0) in the next X (in our case 3) minutes


```python
def create_target_variable(df, future_minutes=3):
    """
    Creates a target variable indicating if the price goes up or down in the specified future timeframe.
    1 indicates an increase, and 0 indicates a decrease or no change.
    """
    # Shift the closing price by the specified future minutes
    df['Future_Close'] = df['CLOSE'].shift(-future_minutes)
    
    # Determine if the future closing price is higher than the current closing price
    df['Target'] = (df['Future_Close'] > df['CLOSE']).astype(int)
    
    # Drop the last 'future_minutes' rows as they won't have a future price
    df = df.dropna(subset=['Target'])
    
    return df

# Apply the function to your DataFrame
df = create_target_variable(df)
# Save the DataFrame to a new CSV file
df.to_csv('/Users/ronangrant/Vault/training_data/indicators/indicators_with_ouput.csv', index=False)

```

### Removes holidays and other dates


```python
import pandas as pd
from datetime import datetime

# Load your CSV file
df = pd.read_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_removed_dates.csv')

# Convert 'datetime' to datetime object
df['datetime'] = pd.to_datetime(df['datetime'])

# Dates to remove
dates_to_remove = [
    "2023-12-25", "2023-11-16", "2023-10-31", "2022-12-08", 
    "2023-03-29", "2023-05-17", "2023-07-11", "2023-08-02", 
    "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", 
    "2023-06-19", "2023-07-04"
]
# Convert strings to datetime objects
dates_to_remove = [pd.to_datetime(date).date() for date in dates_to_remove]

# Remove the rows with these dates
df = df[~df['datetime'].dt.date.isin(dates_to_remove)]

# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_removed_dates2.csv', index=False)

```

### Checks if there are any values RSI that are 0 (Because the days that do have them are usualy holidays)


```python
import pandas as pd

# Load your CSV file
df = pd.read_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_removed_dates2.csv')

# Filter the DataFrame for rows where RSI is 0
rsi_zero_df = df[df['RSI_14'] == 0]

# Count how many RSI values are 0
rsi_zero_count = len(rsi_zero_df)

# Print the count
print(f"Number of RSI values that are 0: {rsi_zero_count}")

# If you want to see the dates as well
if rsi_zero_count > 0:
    print("Dates with RSI value of 0:")
    print(rsi_zero_df['datetime'])

```

### Adds a day of the week column 


```python
import pandas as pd

# Load your CSV file
df = pd.read_csv('/Users/ronangrant/Vault/training_data/indicators/updated_data_removed_dates2.csv')

# Convert 'datetime' to datetime object
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract and add the time column (hour and minute)
df['time'] = df['datetime'].dt.strftime('%H:%M')

# Extract and add the day of the week column
# The day is represented as an integer (Monday=0, Sunday=6)
df['day_of_week'] = df['datetime'].dt.dayofweek

# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/ronangrant/Vault/training_data/updated_data_with_time_and_day.csv', index=False)

```

### Check for missing values


```python
import pandas as pd

# Load your CSV file
df = pd.read_csv('/Users/ronangrant/Vault/training_data/updated_data_with_new_indicators3.csv')


# Check for missing values in the DataFrame
missing_values = df.isnull().sum()

# Print the count of missing values for each column
print("Count of missing values in each column:")
print(missing_values)

# Optional: Save the updated DataFrame to a new CSV file
df.to_csv('/Users/ronangrant/Vault/training_data/updated_data_no_unneeded_columns.csv', index=False)

```

    Count of missing values in each column:
    OPEN                0
    HIGH                0
    LOW                 0
    CLOSE               0
    VOL                 0
    SMA_50              0
    EMA_50              0
    RSI_14              0
    MACD                0
    MACD_Signal         0
    SMA_20              0
    Bollinger_Upper     0
    Bollinger_Lower     0
    Target              0
    time                0
    day_of_week         0
    %K                  0
    %D                  2
    ATR                13
    dtype: int64


### Look for any possible outliers 


```python
import pandas as pd

# Load your CSV file
df = pd.read_csv('/Users/ronangrant/Vault/training_data/updated_data_no_unneeded_columns.csv')

# Define the columns to check for extreme outliers
columns_to_check = ['VOL', 'RSI_14', 'MACD', 'MACD_Signal']

# Function to detect extreme outliers based on a higher threshold
def detect_extreme_outliers(data, feature, threshold=5):
    mean = data[feature].mean()
    std = data[feature].std()
    outliers = data[(data[feature] < (mean - threshold * std)) | (data[feature] > (mean + threshold * std))]
    return outliers

# Check each column for extreme outliers
for col in columns_to_check:
    extreme_outliers = detect_extreme_outliers(df, col)
    print(f"Extreme outliers in '{col}': {len(extreme_outliers)}")
    if len(extreme_outliers) > 0:
        print("Example outliers:")
        print(extreme_outliers.head())  # Print a few examples

        # Ask the user whether to delete these outliers
        user_input = input(f"Do you want to delete extreme outliers in '{col}'? (yes/no): ")
        if user_input.lower() == 'yes':
            df = df.drop(extreme_outliers.index)  # Drop the outliers

# Optional: Save the updated DataFrame to a new CSV file
df.to_csv('/Users/ronangrant/Vault/training_data/data_with_extreme_outliers_removed.csv', index=False)

```

    Extreme outliers in 'VOL': 1466
    Example outliers:
            OPEN     HIGH      LOW    CLOSE    VOL   SMA_50   EMA_50    RSI_14  \
    584  1.04288  1.04294  1.04179  1.04210   5781  1.04465  1.04437  18.83657   
    585  1.04211  1.04212  1.04140  1.04201   6021  1.04459  1.04427  18.68132   
    586  1.04195  1.04260  1.04170  1.04219   6128  1.04453  1.04419  23.88889   
    589  1.04153  1.04177  1.03998  1.04047   9777  1.04429  1.04385  16.42105   
    590  1.04052  1.04099  1.03958  1.03964  10968  1.04418  1.04368  13.82406   
    
            MACD  MACD_Signal   SMA_20  Bollinger_Upper  Bollinger_Lower  Target  \
    584 -0.00049     -0.00037  1.04387          1.04504          1.04269       0   
    585 -0.00057     -0.00041  1.04375          1.04517          1.04233       0   
    586 -0.00062     -0.00045  1.04364          1.04519          1.04209       0   
    589 -0.00087     -0.00062  1.04314          1.04537          1.04092       0   
    590 -0.00102     -0.00070  1.04291          1.04555          1.04026       1   
    
          time  day_of_week  
    584  07:52            3  
    585  07:53            3  
    586  07:54            3  
    589  07:57            3  
    590  07:58            3  


    Do you want to delete extreme outliers in 'VOL'? (yes/no):  no


    Extreme outliers in 'RSI_14': 0
    Extreme outliers in 'MACD': 1324
    Example outliers:
            OPEN     HIGH      LOW    CLOSE    VOL   SMA_50   EMA_50    RSI_14  \
    589  1.04153  1.04177  1.03998  1.04047   9777  1.04429  1.04385  16.42105   
    590  1.04052  1.04099  1.03958  1.03964  10968  1.04418  1.04368  13.82406   
    591  1.03962  1.04048  1.03920  1.04028  10793  1.04409  1.04355  21.31148   
    592  1.04034  1.04075  1.03977  1.03980  11810  1.04399  1.04340  19.51220   
    593  1.03988  1.04047  1.03906  1.04032  10650  1.04391  1.04328  22.91971   
    
            MACD  MACD_Signal   SMA_20  Bollinger_Upper  Bollinger_Lower  Target  \
    589 -0.00087     -0.00062  1.04314          1.04537          1.04092       0   
    590 -0.00102     -0.00070  1.04291          1.04555          1.04026       1   
    591 -0.00107     -0.00077  1.04271          1.04552          1.03990       1   
    592 -0.00114     -0.00085  1.04250          1.04551          1.03948       1   
    593 -0.00114     -0.00091  1.04232          1.04543          1.03922       1   
    
          time  day_of_week  
    589  07:57            3  
    590  07:58            3  
    591  07:59            3  
    592  08:00            3  
    593  08:01            3  


    Do you want to delete extreme outliers in 'MACD'? (yes/no):  no


    Extreme outliers in 'MACD_Signal': 1395
    Example outliers:
            OPEN     HIGH      LOW    CLOSE    VOL   SMA_50   EMA_50    RSI_14  \
    592  1.04034  1.04075  1.03977  1.03980  11810  1.04399  1.04340  19.51220   
    593  1.03988  1.04047  1.03906  1.04032  10650  1.04391  1.04328  22.91971   
    594  1.04028  1.04089  1.04000  1.04071   6874  1.04383  1.04318  27.92023   
    595  1.04071  1.04107  1.04039  1.04089   4341  1.04375  1.04309  27.40316   
    596  1.04091  1.04135  1.04069  1.04134   3930  1.04368  1.04302  34.85968   
    
            MACD  MACD_Signal   SMA_20  Bollinger_Upper  Bollinger_Lower  Target  \
    592 -0.00114     -0.00085  1.04250          1.04551          1.03948       1   
    593 -0.00114     -0.00091  1.04232          1.04543          1.03922       1   
    594 -0.00110     -0.00094  1.04218          1.04530          1.03906       1   
    595 -0.00103     -0.00096  1.04204          1.04513          1.03895       1   
    596 -0.00094     -0.00096  1.04193          1.04493          1.03892       1   
    
          time  day_of_week  
    592  08:00            3  
    593  08:01            3  
    594  08:02            3  
    595  08:03            3  
    596  08:04            3  


    Do you want to delete extreme outliers in 'MACD_Signal'? (yes/no):  no


### Random Forest Classifier - 3 Minutes


```python
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
file_path = '/Users/ronangrant/Vault/training_data/updated_data_no_unneeded_columns.csv'
df = pd.read_csv(file_path)

# Convert 'time' from string to numerical values (hours and minutes)
df['hour'] = pd.to_datetime(df['time']).dt.hour
df['minute'] = pd.to_datetime(df['time']).dt.minute

# Drop the original 'time' column
df.drop('time', axis=1, inplace=True)

# Define features (X) and target (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy}")

```

    /var/folders/t8/zyslw4cj6rz9td3j918w22_h0000gn/T/ipykernel_57019/2010147340.py:11: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      df['hour'] = pd.to_datetime(df['time']).dt.hour
    /var/folders/t8/zyslw4cj6rz9td3j918w22_h0000gn/T/ipykernel_57019/2010147340.py:12: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      df['minute'] = pd.to_datetime(df['time']).dt.minute


    Random Forest Classifier Accuracy: 0.6731189667350577


### Neural Network - When I ran this, it was with a eariler dataset. And I did not really try much. This is with the 3 minute target


```python
!pip install tensorflow-macos
!pip install tensorflow-metal

```

    Requirement already satisfied: tensorflow-macos in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (2.15.0)
    Requirement already satisfied: absl-py>=1.0.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (2.0.0)
    Requirement already satisfied: astunparse>=1.6.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (1.6.3)
    Requirement already satisfied: flatbuffers>=23.5.26 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (23.5.26)
    Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (0.5.4)
    Requirement already satisfied: google-pasta>=0.1.1 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (3.10.0)
    Requirement already satisfied: libclang>=13.0.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (16.0.6)
    Requirement already satisfied: ml-dtypes~=0.2.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (0.2.0)
    Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (1.26.2)
    Requirement already satisfied: opt-einsum>=2.3.2 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (3.3.0)
    Requirement already satisfied: packaging in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (23.2)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (4.23.4)
    Requirement already satisfied: setuptools in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (68.2.2)
    Requirement already satisfied: six>=1.12.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (2.4.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (4.9.0)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (1.14.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (0.34.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (1.60.0)
    Requirement already satisfied: tensorboard<2.16,>=2.15 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (2.15.1)
    Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (2.15.0)
    Requirement already satisfied: keras<2.16,>=2.15.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-macos) (2.15.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from astunparse>=1.6.0->tensorflow-macos) (0.41.2)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorboard<2.16,>=2.15->tensorflow-macos) (2.25.2)
    Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorboard<2.16,>=2.15->tensorflow-macos) (1.2.0)
    Requirement already satisfied: markdown>=2.6.8 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorboard<2.16,>=2.15->tensorflow-macos) (3.5.1)
    Requirement already satisfied: requests<3,>=2.21.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorboard<2.16,>=2.15->tensorflow-macos) (2.31.0)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorboard<2.16,>=2.15->tensorflow-macos) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorboard<2.16,>=2.15->tensorflow-macos) (3.0.1)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-macos) (5.3.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-macos) (0.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-macos) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow-macos) (1.3.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-macos) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-macos) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-macos) (2.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-macos) (2023.11.17)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow-macos) (2.1.3)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-macos) (0.5.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow-macos) (3.2.2)
    Collecting tensorflow-metal
      Downloading tensorflow_metal-1.1.0-cp311-cp311-macosx_12_0_arm64.whl.metadata (1.2 kB)
    Requirement already satisfied: wheel~=0.35 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-metal) (0.41.2)
    Requirement already satisfied: six>=1.15.0 in /Users/ronangrant/anaconda3/envs/TradingBot/lib/python3.11/site-packages (from tensorflow-metal) (1.16.0)
    Downloading tensorflow_metal-1.1.0-cp311-cp311-macosx_12_0_arm64.whl (1.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.4/1.4 MB[0m [31m9.0 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hInstalling collected packages: tensorflow-metal
    Successfully installed tensorflow-metal-1.1.0



```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the data
file_path = '/Users/ronangrant/Vault/training_data/cleaned_data2.csv'
df = pd.read_csv(file_path)

# Convert 'time' to numerical values (if you haven't already)
df['hour'] = pd.to_datetime(df['time']).dt.hour
df['minute'] = pd.to_datetime(df['time']).dt.minute
df.drop('time', axis=1, inplace=True)

# Define features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural network architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Neural Network Accuracy: {accuracy}")

```

    /var/folders/t8/zyslw4cj6rz9td3j918w22_h0000gn/T/ipykernel_57019/3896504100.py:13: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      df['hour'] = pd.to_datetime(df['time']).dt.hour
    /var/folders/t8/zyslw4cj6rz9td3j918w22_h0000gn/T/ipykernel_57019/3896504100.py:14: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      df['minute'] = pd.to_datetime(df['time']).dt.minute
    WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.


    Epoch 1/50
    5946/5946 [==============================] - 2s 327us/step - loss: 0.6885 - accuracy: 0.5329 - val_loss: 0.6892 - val_accuracy: 0.5322
    Epoch 2/50
    5946/5946 [==============================] - 2s 316us/step - loss: 0.6857 - accuracy: 0.5391 - val_loss: 0.6866 - val_accuracy: 0.5378
    Epoch 3/50
    5946/5946 [==============================] - 2s 315us/step - loss: 0.6851 - accuracy: 0.5410 - val_loss: 0.6868 - val_accuracy: 0.5365
    Epoch 4/50
    5946/5946 [==============================] - 2s 323us/step - loss: 0.6847 - accuracy: 0.5416 - val_loss: 0.6867 - val_accuracy: 0.5362
    Epoch 5/50
    5946/5946 [==============================] - 2s 316us/step - loss: 0.6845 - accuracy: 0.5436 - val_loss: 0.6871 - val_accuracy: 0.5365
    Epoch 6/50
    5946/5946 [==============================] - 2s 313us/step - loss: 0.6843 - accuracy: 0.5443 - val_loss: 0.6865 - val_accuracy: 0.5366
    Epoch 7/50
    5946/5946 [==============================] - 2s 322us/step - loss: 0.6840 - accuracy: 0.5434 - val_loss: 0.6862 - val_accuracy: 0.5402
    Epoch 8/50
    5946/5946 [==============================] - 2s 313us/step - loss: 0.6838 - accuracy: 0.5453 - val_loss: 0.6862 - val_accuracy: 0.5378
    Epoch 9/50
    5946/5946 [==============================] - 2s 311us/step - loss: 0.6835 - accuracy: 0.5459 - val_loss: 0.6869 - val_accuracy: 0.5379
    Epoch 10/50
    5946/5946 [==============================] - 2s 313us/step - loss: 0.6834 - accuracy: 0.5468 - val_loss: 0.6868 - val_accuracy: 0.5381
    Epoch 11/50
    5946/5946 [==============================] - 2s 310us/step - loss: 0.6832 - accuracy: 0.5471 - val_loss: 0.6867 - val_accuracy: 0.5392
    Epoch 12/50
    5946/5946 [==============================] - 2s 311us/step - loss: 0.6831 - accuracy: 0.5483 - val_loss: 0.6861 - val_accuracy: 0.5391
    Epoch 13/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6830 - accuracy: 0.5483 - val_loss: 0.6864 - val_accuracy: 0.5394
    Epoch 14/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6827 - accuracy: 0.5489 - val_loss: 0.6865 - val_accuracy: 0.5397
    Epoch 15/50
    5946/5946 [==============================] - 2s 311us/step - loss: 0.6824 - accuracy: 0.5502 - val_loss: 0.6863 - val_accuracy: 0.5381
    Epoch 16/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6823 - accuracy: 0.5506 - val_loss: 0.6877 - val_accuracy: 0.5341
    Epoch 17/50
    5946/5946 [==============================] - 2s 316us/step - loss: 0.6823 - accuracy: 0.5496 - val_loss: 0.6864 - val_accuracy: 0.5389
    Epoch 18/50
    5946/5946 [==============================] - 2s 313us/step - loss: 0.6820 - accuracy: 0.5513 - val_loss: 0.6868 - val_accuracy: 0.5396
    Epoch 19/50
    5946/5946 [==============================] - 2s 311us/step - loss: 0.6817 - accuracy: 0.5511 - val_loss: 0.6867 - val_accuracy: 0.5394
    Epoch 20/50
    5946/5946 [==============================] - 2s 315us/step - loss: 0.6816 - accuracy: 0.5517 - val_loss: 0.6867 - val_accuracy: 0.5409
    Epoch 21/50
    5946/5946 [==============================] - 2s 315us/step - loss: 0.6813 - accuracy: 0.5532 - val_loss: 0.6862 - val_accuracy: 0.5400
    Epoch 22/50
    5946/5946 [==============================] - 2s 310us/step - loss: 0.6811 - accuracy: 0.5520 - val_loss: 0.6861 - val_accuracy: 0.5392
    Epoch 23/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6810 - accuracy: 0.5533 - val_loss: 0.6883 - val_accuracy: 0.5396
    Epoch 24/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6807 - accuracy: 0.5534 - val_loss: 0.6884 - val_accuracy: 0.5422
    Epoch 25/50
    5946/5946 [==============================] - 2s 310us/step - loss: 0.6806 - accuracy: 0.5548 - val_loss: 0.6862 - val_accuracy: 0.5417
    Epoch 26/50
    5946/5946 [==============================] - 2s 311us/step - loss: 0.6803 - accuracy: 0.5560 - val_loss: 0.6866 - val_accuracy: 0.5414
    Epoch 27/50
    5946/5946 [==============================] - 2s 314us/step - loss: 0.6802 - accuracy: 0.5563 - val_loss: 0.6869 - val_accuracy: 0.5399
    Epoch 28/50
    5946/5946 [==============================] - 2s 311us/step - loss: 0.6800 - accuracy: 0.5568 - val_loss: 0.6881 - val_accuracy: 0.5381
    Epoch 29/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6799 - accuracy: 0.5578 - val_loss: 0.6871 - val_accuracy: 0.5416
    Epoch 30/50
    5946/5946 [==============================] - 2s 314us/step - loss: 0.6797 - accuracy: 0.5577 - val_loss: 0.6890 - val_accuracy: 0.5377
    Epoch 31/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6794 - accuracy: 0.5577 - val_loss: 0.6885 - val_accuracy: 0.5397
    Epoch 32/50
    5946/5946 [==============================] - 2s 325us/step - loss: 0.6793 - accuracy: 0.5594 - val_loss: 0.6876 - val_accuracy: 0.5397
    Epoch 33/50
    5946/5946 [==============================] - 2s 317us/step - loss: 0.6792 - accuracy: 0.5591 - val_loss: 0.6877 - val_accuracy: 0.5393
    Epoch 34/50
    5946/5946 [==============================] - 2s 324us/step - loss: 0.6790 - accuracy: 0.5584 - val_loss: 0.6868 - val_accuracy: 0.5437
    Epoch 35/50
    5946/5946 [==============================] - 2s 315us/step - loss: 0.6787 - accuracy: 0.5595 - val_loss: 0.6872 - val_accuracy: 0.5432
    Epoch 36/50
    5946/5946 [==============================] - 2s 320us/step - loss: 0.6786 - accuracy: 0.5591 - val_loss: 0.6879 - val_accuracy: 0.5400
    Epoch 37/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6785 - accuracy: 0.5595 - val_loss: 0.6881 - val_accuracy: 0.5422
    Epoch 38/50
    5946/5946 [==============================] - 2s 314us/step - loss: 0.6783 - accuracy: 0.5603 - val_loss: 0.6870 - val_accuracy: 0.5425
    Epoch 39/50
    5946/5946 [==============================] - 2s 314us/step - loss: 0.6781 - accuracy: 0.5631 - val_loss: 0.6876 - val_accuracy: 0.5411
    Epoch 40/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6780 - accuracy: 0.5614 - val_loss: 0.6873 - val_accuracy: 0.5435
    Epoch 41/50
    5946/5946 [==============================] - 2s 320us/step - loss: 0.6779 - accuracy: 0.5615 - val_loss: 0.6867 - val_accuracy: 0.5400
    Epoch 42/50
    5946/5946 [==============================] - 2s 320us/step - loss: 0.6777 - accuracy: 0.5621 - val_loss: 0.6867 - val_accuracy: 0.5436
    Epoch 43/50
    5946/5946 [==============================] - 2s 328us/step - loss: 0.6774 - accuracy: 0.5628 - val_loss: 0.6867 - val_accuracy: 0.5418
    Epoch 44/50
    5946/5946 [==============================] - 2s 326us/step - loss: 0.6773 - accuracy: 0.5628 - val_loss: 0.6872 - val_accuracy: 0.5408
    Epoch 45/50
    5946/5946 [==============================] - 2s 318us/step - loss: 0.6774 - accuracy: 0.5626 - val_loss: 0.6872 - val_accuracy: 0.5412
    Epoch 46/50
    5946/5946 [==============================] - 2s 312us/step - loss: 0.6771 - accuracy: 0.5617 - val_loss: 0.6879 - val_accuracy: 0.5395
    Epoch 47/50
    5946/5946 [==============================] - 2s 315us/step - loss: 0.6771 - accuracy: 0.5626 - val_loss: 0.6875 - val_accuracy: 0.5438
    Epoch 48/50
    5946/5946 [==============================] - 2s 308us/step - loss: 0.6769 - accuracy: 0.5628 - val_loss: 0.6864 - val_accuracy: 0.5444
    Epoch 49/50
    5946/5946 [==============================] - 2s 311us/step - loss: 0.6768 - accuracy: 0.5632 - val_loss: 0.6866 - val_accuracy: 0.5445
    Epoch 50/50
    5946/5946 [==============================] - 2s 313us/step - loss: 0.6767 - accuracy: 0.5640 - val_loss: 0.6874 - val_accuracy: 0.5402
    1859/1859 [==============================] - 0s 200us/step - loss: 0.6864 - accuracy: 0.5445
    Neural Network Accuracy: 0.5444668531417847



```python

```

### XGBoost Classifier - When I ran this, it was with a eariler dataset. And I did not really try much. This is with the 3 minute target


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the data
file_path = '/Users/ronangrant/Vault/training_data/cleaned_data2.csv'
df = pd.read_csv(file_path)

# Convert 'time' from string to numerical values (hours and minutes)
df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
df['minute'] = pd.to_datetime(df['time'], format='%H:%M').dt.minute
df.drop('time', axis=1, inplace=True)

# Define features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier with updated hyperparameters
xgb_classifier = XGBClassifier(
    n_estimators=1000,  # Increased number of trees
    max_depth=80,        # Slightly deeper trees
    learning_rate=0.01, # Lower learning rate
    min_child_weight=5, # Increase this to control over-fitting
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=.6,            # Increase gamma to reduce complexity
    early_stopping_rounds=20, # Early stopping rounds
    eval_metric="logloss",   # Evaluation metric
    random_state=42
)

# Train the classifier with evaluation set for early stopping
xgb_classifier.fit(
    X_train, y_train, 
    eval_set=[(X_test, y_test)], 
    verbose=True
)

# Make predictions
y_pred = xgb_classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Classifier Accuracy: {accuracy}")
```

    [0]	validation_0-logloss:0.69253
    [1]	validation_0-logloss:0.69228
    [2]	validation_0-logloss:0.69202
    [3]	validation_0-logloss:0.69176
    [4]	validation_0-logloss:0.69153
    [5]	validation_0-logloss:0.69126
    [6]	validation_0-logloss:0.69101
    [7]	validation_0-logloss:0.69078
    [8]	validation_0-logloss:0.69055
    [9]	validation_0-logloss:0.69031
    [10]	validation_0-logloss:0.69003
    [11]	validation_0-logloss:0.68981
    [12]	validation_0-logloss:0.68959
    [13]	validation_0-logloss:0.68937
    [14]	validation_0-logloss:0.68914
    [15]	validation_0-logloss:0.68882
    [16]	validation_0-logloss:0.68858
    [17]	validation_0-logloss:0.68829
    [18]	validation_0-logloss:0.68802
    [19]	validation_0-logloss:0.68779
    [20]	validation_0-logloss:0.68760
    [21]	validation_0-logloss:0.68741
    [22]	validation_0-logloss:0.68720
    [23]	validation_0-logloss:0.68699
    [24]	validation_0-logloss:0.68675
    [25]	validation_0-logloss:0.68657
    [26]	validation_0-logloss:0.68633
    [27]	validation_0-logloss:0.68609
    [28]	validation_0-logloss:0.68589
    [29]	validation_0-logloss:0.68572
    [30]	validation_0-logloss:0.68555
    [31]	validation_0-logloss:0.68535
    [32]	validation_0-logloss:0.68518
    [33]	validation_0-logloss:0.68493
    [34]	validation_0-logloss:0.68462
    [35]	validation_0-logloss:0.68448
    [36]	validation_0-logloss:0.68431
    [37]	validation_0-logloss:0.68409
    [38]	validation_0-logloss:0.68393
    [39]	validation_0-logloss:0.68367
    [40]	validation_0-logloss:0.68345
    [41]	validation_0-logloss:0.68333
    [42]	validation_0-logloss:0.68317
    [43]	validation_0-logloss:0.68289
    [44]	validation_0-logloss:0.68255
    [45]	validation_0-logloss:0.68230
    [46]	validation_0-logloss:0.68216
    [47]	validation_0-logloss:0.68204
    [48]	validation_0-logloss:0.68190
    [49]	validation_0-logloss:0.68176
    [50]	validation_0-logloss:0.68163
    [51]	validation_0-logloss:0.68152
    [52]	validation_0-logloss:0.68132
    [53]	validation_0-logloss:0.68114
    [54]	validation_0-logloss:0.68099
    [55]	validation_0-logloss:0.68084
    [56]	validation_0-logloss:0.68067
    [57]	validation_0-logloss:0.68052
    [58]	validation_0-logloss:0.68031
    [59]	validation_0-logloss:0.68011
    [60]	validation_0-logloss:0.67988
    [61]	validation_0-logloss:0.67974
    [62]	validation_0-logloss:0.67962
    [63]	validation_0-logloss:0.67938
    [64]	validation_0-logloss:0.67925
    [65]	validation_0-logloss:0.67913
    [66]	validation_0-logloss:0.67902
    [67]	validation_0-logloss:0.67886
    [68]	validation_0-logloss:0.67879
    [69]	validation_0-logloss:0.67868
    [70]	validation_0-logloss:0.67854
    [71]	validation_0-logloss:0.67843
    [72]	validation_0-logloss:0.67827
    [73]	validation_0-logloss:0.67813
    [74]	validation_0-logloss:0.67797
    [75]	validation_0-logloss:0.67789
    [76]	validation_0-logloss:0.67768
    [77]	validation_0-logloss:0.67756
    [78]	validation_0-logloss:0.67735
    [79]	validation_0-logloss:0.67708
    [80]	validation_0-logloss:0.67696
    [81]	validation_0-logloss:0.67686
    [82]	validation_0-logloss:0.67677
    [83]	validation_0-logloss:0.67667
    [84]	validation_0-logloss:0.67655
    [85]	validation_0-logloss:0.67636
    [86]	validation_0-logloss:0.67629
    [87]	validation_0-logloss:0.67622
    [88]	validation_0-logloss:0.67616
    [89]	validation_0-logloss:0.67607
    [90]	validation_0-logloss:0.67597
    [91]	validation_0-logloss:0.67590
    [92]	validation_0-logloss:0.67579
    [93]	validation_0-logloss:0.67571
    [94]	validation_0-logloss:0.67561
    [95]	validation_0-logloss:0.67552
    [96]	validation_0-logloss:0.67545
    [97]	validation_0-logloss:0.67526
    [98]	validation_0-logloss:0.67509
    [99]	validation_0-logloss:0.67493
    [100]	validation_0-logloss:0.67484
    [101]	validation_0-logloss:0.67474
    [102]	validation_0-logloss:0.67460
    [103]	validation_0-logloss:0.67442
    [104]	validation_0-logloss:0.67421
    [105]	validation_0-logloss:0.67400
    [106]	validation_0-logloss:0.67384
    [107]	validation_0-logloss:0.67366
    [108]	validation_0-logloss:0.67344
    [109]	validation_0-logloss:0.67332
    [110]	validation_0-logloss:0.67317
    [111]	validation_0-logloss:0.67309
    [112]	validation_0-logloss:0.67303
    [113]	validation_0-logloss:0.67298
    [114]	validation_0-logloss:0.67278
    [115]	validation_0-logloss:0.67271
    [116]	validation_0-logloss:0.67263
    [117]	validation_0-logloss:0.67256
    [118]	validation_0-logloss:0.67250
    [119]	validation_0-logloss:0.67236
    [120]	validation_0-logloss:0.67230
    [121]	validation_0-logloss:0.67221
    [122]	validation_0-logloss:0.67192
    [123]	validation_0-logloss:0.67172
    [124]	validation_0-logloss:0.67166
    [125]	validation_0-logloss:0.67157
    [126]	validation_0-logloss:0.67145
    [127]	validation_0-logloss:0.67140
    [128]	validation_0-logloss:0.67129
    [129]	validation_0-logloss:0.67121
    [130]	validation_0-logloss:0.67113
    [131]	validation_0-logloss:0.67107
    [132]	validation_0-logloss:0.67101
    [133]	validation_0-logloss:0.67095
    [134]	validation_0-logloss:0.67088
    [135]	validation_0-logloss:0.67081
    [136]	validation_0-logloss:0.67067
    [137]	validation_0-logloss:0.67064
    [138]	validation_0-logloss:0.67061
    [139]	validation_0-logloss:0.67057
    [140]	validation_0-logloss:0.67053
    [141]	validation_0-logloss:0.67040
    [142]	validation_0-logloss:0.67032
    [143]	validation_0-logloss:0.67016
    [144]	validation_0-logloss:0.67013
    [145]	validation_0-logloss:0.66995
    [146]	validation_0-logloss:0.66993
    [147]	validation_0-logloss:0.66978
    [148]	validation_0-logloss:0.66969
    [149]	validation_0-logloss:0.66964
    [150]	validation_0-logloss:0.66945
    [151]	validation_0-logloss:0.66940
    [152]	validation_0-logloss:0.66932
    [153]	validation_0-logloss:0.66925
    [154]	validation_0-logloss:0.66918
    [155]	validation_0-logloss:0.66910
    [156]	validation_0-logloss:0.66905
    [157]	validation_0-logloss:0.66899
    [158]	validation_0-logloss:0.66893
    [159]	validation_0-logloss:0.66888
    [160]	validation_0-logloss:0.66884
    [161]	validation_0-logloss:0.66880
    [162]	validation_0-logloss:0.66874
    [163]	validation_0-logloss:0.66858
    [164]	validation_0-logloss:0.66855
    [165]	validation_0-logloss:0.66852
    [166]	validation_0-logloss:0.66825
    [167]	validation_0-logloss:0.66818
    [168]	validation_0-logloss:0.66812
    [169]	validation_0-logloss:0.66800
    [170]	validation_0-logloss:0.66797
    [171]	validation_0-logloss:0.66794
    [172]	validation_0-logloss:0.66788
    [173]	validation_0-logloss:0.66783
    [174]	validation_0-logloss:0.66778
    [175]	validation_0-logloss:0.66756
    [176]	validation_0-logloss:0.66753
    [177]	validation_0-logloss:0.66752
    [178]	validation_0-logloss:0.66732
    [179]	validation_0-logloss:0.66723
    [180]	validation_0-logloss:0.66719
    [181]	validation_0-logloss:0.66704
    [182]	validation_0-logloss:0.66699
    [183]	validation_0-logloss:0.66695
    [184]	validation_0-logloss:0.66683
    [185]	validation_0-logloss:0.66675
    [186]	validation_0-logloss:0.66667
    [187]	validation_0-logloss:0.66658
    [188]	validation_0-logloss:0.66657
    [189]	validation_0-logloss:0.66643
    [190]	validation_0-logloss:0.66628
    [191]	validation_0-logloss:0.66613
    [192]	validation_0-logloss:0.66601
    [193]	validation_0-logloss:0.66579
    [194]	validation_0-logloss:0.66573
    [195]	validation_0-logloss:0.66570
    [196]	validation_0-logloss:0.66566
    [197]	validation_0-logloss:0.66562
    [198]	validation_0-logloss:0.66559
    [199]	validation_0-logloss:0.66551
    [200]	validation_0-logloss:0.66549
    [201]	validation_0-logloss:0.66534
    [202]	validation_0-logloss:0.66533
    [203]	validation_0-logloss:0.66531
    [204]	validation_0-logloss:0.66527
    [205]	validation_0-logloss:0.66522
    [206]	validation_0-logloss:0.66520
    [207]	validation_0-logloss:0.66516
    [208]	validation_0-logloss:0.66514
    [209]	validation_0-logloss:0.66510
    [210]	validation_0-logloss:0.66504
    [211]	validation_0-logloss:0.66501
    [212]	validation_0-logloss:0.66482
    [213]	validation_0-logloss:0.66481
    [214]	validation_0-logloss:0.66476
    [215]	validation_0-logloss:0.66471
    [216]	validation_0-logloss:0.66460
    [217]	validation_0-logloss:0.66448
    [218]	validation_0-logloss:0.66442
    [219]	validation_0-logloss:0.66439
    [220]	validation_0-logloss:0.66432
    [221]	validation_0-logloss:0.66429
    [222]	validation_0-logloss:0.66428
    [223]	validation_0-logloss:0.66429
    [224]	validation_0-logloss:0.66416
    [225]	validation_0-logloss:0.66415
    [226]	validation_0-logloss:0.66410
    [227]	validation_0-logloss:0.66408
    [228]	validation_0-logloss:0.66400
    [229]	validation_0-logloss:0.66395
    [230]	validation_0-logloss:0.66381
    [231]	validation_0-logloss:0.66374
    [232]	validation_0-logloss:0.66373
    [233]	validation_0-logloss:0.66359
    [234]	validation_0-logloss:0.66354
    [235]	validation_0-logloss:0.66344
    [236]	validation_0-logloss:0.66341
    [237]	validation_0-logloss:0.66335
    [238]	validation_0-logloss:0.66332
    [239]	validation_0-logloss:0.66332
    [240]	validation_0-logloss:0.66329
    [241]	validation_0-logloss:0.66326
    [242]	validation_0-logloss:0.66314
    [243]	validation_0-logloss:0.66312
    [244]	validation_0-logloss:0.66306
    [245]	validation_0-logloss:0.66300
    [246]	validation_0-logloss:0.66296
    [247]	validation_0-logloss:0.66294
    [248]	validation_0-logloss:0.66287
    [249]	validation_0-logloss:0.66288
    [250]	validation_0-logloss:0.66287
    [251]	validation_0-logloss:0.66271
    [252]	validation_0-logloss:0.66268
    [253]	validation_0-logloss:0.66265
    [254]	validation_0-logloss:0.66260
    [255]	validation_0-logloss:0.66261
    [256]	validation_0-logloss:0.66254
    [257]	validation_0-logloss:0.66253
    [258]	validation_0-logloss:0.66238
    [259]	validation_0-logloss:0.66233
    [260]	validation_0-logloss:0.66220
    [261]	validation_0-logloss:0.66217
    [262]	validation_0-logloss:0.66213
    [263]	validation_0-logloss:0.66211
    [264]	validation_0-logloss:0.66209
    [265]	validation_0-logloss:0.66204
    [266]	validation_0-logloss:0.66192
    [267]	validation_0-logloss:0.66176
    [268]	validation_0-logloss:0.66172
    [269]	validation_0-logloss:0.66169
    [270]	validation_0-logloss:0.66156
    [271]	validation_0-logloss:0.66153
    [272]	validation_0-logloss:0.66152
    [273]	validation_0-logloss:0.66150
    [274]	validation_0-logloss:0.66138
    [275]	validation_0-logloss:0.66124
    [276]	validation_0-logloss:0.66123
    [277]	validation_0-logloss:0.66121
    [278]	validation_0-logloss:0.66108
    [279]	validation_0-logloss:0.66106
    [280]	validation_0-logloss:0.66099
    [281]	validation_0-logloss:0.66095
    [282]	validation_0-logloss:0.66095
    [283]	validation_0-logloss:0.66080
    [284]	validation_0-logloss:0.66079
    [285]	validation_0-logloss:0.66069
    [286]	validation_0-logloss:0.66067
    [287]	validation_0-logloss:0.66069
    [288]	validation_0-logloss:0.66055
    [289]	validation_0-logloss:0.66050
    [290]	validation_0-logloss:0.66044
    [291]	validation_0-logloss:0.66043
    [292]	validation_0-logloss:0.66042
    [293]	validation_0-logloss:0.66030
    [294]	validation_0-logloss:0.66012
    [295]	validation_0-logloss:0.66013
    [296]	validation_0-logloss:0.66012
    [297]	validation_0-logloss:0.66010
    [298]	validation_0-logloss:0.66004
    [299]	validation_0-logloss:0.66001
    [300]	validation_0-logloss:0.65998
    [301]	validation_0-logloss:0.65988
    [302]	validation_0-logloss:0.65987
    [303]	validation_0-logloss:0.65986
    [304]	validation_0-logloss:0.65983
    [305]	validation_0-logloss:0.65981
    [306]	validation_0-logloss:0.65975
    [307]	validation_0-logloss:0.65966
    [308]	validation_0-logloss:0.65966
    [309]	validation_0-logloss:0.65948
    [310]	validation_0-logloss:0.65941
    [311]	validation_0-logloss:0.65924
    [312]	validation_0-logloss:0.65925
    [313]	validation_0-logloss:0.65923
    [314]	validation_0-logloss:0.65923
    [315]	validation_0-logloss:0.65922
    [316]	validation_0-logloss:0.65921
    [317]	validation_0-logloss:0.65917
    [318]	validation_0-logloss:0.65918
    [319]	validation_0-logloss:0.65917
    [320]	validation_0-logloss:0.65914
    [321]	validation_0-logloss:0.65902
    [322]	validation_0-logloss:0.65885
    [323]	validation_0-logloss:0.65882
    [324]	validation_0-logloss:0.65873
    [325]	validation_0-logloss:0.65872
    [326]	validation_0-logloss:0.65869
    [327]	validation_0-logloss:0.65861
    [328]	validation_0-logloss:0.65851
    [329]	validation_0-logloss:0.65847
    [330]	validation_0-logloss:0.65846
    [331]	validation_0-logloss:0.65837
    [332]	validation_0-logloss:0.65834
    [333]	validation_0-logloss:0.65829
    [334]	validation_0-logloss:0.65818
    [335]	validation_0-logloss:0.65812
    [336]	validation_0-logloss:0.65806
    [337]	validation_0-logloss:0.65800
    [338]	validation_0-logloss:0.65789
    [339]	validation_0-logloss:0.65788
    [340]	validation_0-logloss:0.65785
    [341]	validation_0-logloss:0.65779
    [342]	validation_0-logloss:0.65777
    [343]	validation_0-logloss:0.65772
    [344]	validation_0-logloss:0.65770
    [345]	validation_0-logloss:0.65763
    [346]	validation_0-logloss:0.65762
    [347]	validation_0-logloss:0.65758
    [348]	validation_0-logloss:0.65755
    [349]	validation_0-logloss:0.65756
    [350]	validation_0-logloss:0.65745
    [351]	validation_0-logloss:0.65739
    [352]	validation_0-logloss:0.65734
    [353]	validation_0-logloss:0.65730
    [354]	validation_0-logloss:0.65724
    [355]	validation_0-logloss:0.65712
    [356]	validation_0-logloss:0.65712
    [357]	validation_0-logloss:0.65696
    [358]	validation_0-logloss:0.65691
    [359]	validation_0-logloss:0.65685
    [360]	validation_0-logloss:0.65683
    [361]	validation_0-logloss:0.65681
    [362]	validation_0-logloss:0.65682
    [363]	validation_0-logloss:0.65679
    [364]	validation_0-logloss:0.65678
    [365]	validation_0-logloss:0.65668
    [366]	validation_0-logloss:0.65662
    [367]	validation_0-logloss:0.65653
    [368]	validation_0-logloss:0.65649
    [369]	validation_0-logloss:0.65642
    [370]	validation_0-logloss:0.65640
    [371]	validation_0-logloss:0.65641
    [372]	validation_0-logloss:0.65628
    [373]	validation_0-logloss:0.65629
    [374]	validation_0-logloss:0.65628
    [375]	validation_0-logloss:0.65625
    [376]	validation_0-logloss:0.65619
    [377]	validation_0-logloss:0.65620
    [378]	validation_0-logloss:0.65618
    [379]	validation_0-logloss:0.65617
    [380]	validation_0-logloss:0.65614
    [381]	validation_0-logloss:0.65612
    [382]	validation_0-logloss:0.65610
    [383]	validation_0-logloss:0.65604
    [384]	validation_0-logloss:0.65602
    [385]	validation_0-logloss:0.65599
    [386]	validation_0-logloss:0.65582
    [387]	validation_0-logloss:0.65583
    [388]	validation_0-logloss:0.65577
    [389]	validation_0-logloss:0.65573
    [390]	validation_0-logloss:0.65565
    [391]	validation_0-logloss:0.65561
    [392]	validation_0-logloss:0.65562
    [393]	validation_0-logloss:0.65562
    [394]	validation_0-logloss:0.65558
    [395]	validation_0-logloss:0.65555
    [396]	validation_0-logloss:0.65555
    [397]	validation_0-logloss:0.65553
    [398]	validation_0-logloss:0.65555
    [399]	validation_0-logloss:0.65551
    [400]	validation_0-logloss:0.65543
    [401]	validation_0-logloss:0.65540
    [402]	validation_0-logloss:0.65538
    [403]	validation_0-logloss:0.65535
    [404]	validation_0-logloss:0.65533
    [405]	validation_0-logloss:0.65531
    [406]	validation_0-logloss:0.65524
    [407]	validation_0-logloss:0.65519
    [408]	validation_0-logloss:0.65515
    [409]	validation_0-logloss:0.65514
    [410]	validation_0-logloss:0.65514
    [411]	validation_0-logloss:0.65511
    [412]	validation_0-logloss:0.65507
    [413]	validation_0-logloss:0.65499
    [414]	validation_0-logloss:0.65494
    [415]	validation_0-logloss:0.65486
    [416]	validation_0-logloss:0.65483
    [417]	validation_0-logloss:0.65478
    [418]	validation_0-logloss:0.65476
    [419]	validation_0-logloss:0.65472
    [420]	validation_0-logloss:0.65469
    [421]	validation_0-logloss:0.65460
    [422]	validation_0-logloss:0.65456
    [423]	validation_0-logloss:0.65456
    [424]	validation_0-logloss:0.65452
    [425]	validation_0-logloss:0.65447
    [426]	validation_0-logloss:0.65446
    [427]	validation_0-logloss:0.65433
    [428]	validation_0-logloss:0.65433
    [429]	validation_0-logloss:0.65430
    [430]	validation_0-logloss:0.65426
    [431]	validation_0-logloss:0.65425
    [432]	validation_0-logloss:0.65417
    [433]	validation_0-logloss:0.65412
    [434]	validation_0-logloss:0.65403
    [435]	validation_0-logloss:0.65403
    [436]	validation_0-logloss:0.65402
    [437]	validation_0-logloss:0.65396
    [438]	validation_0-logloss:0.65395
    [439]	validation_0-logloss:0.65395
    [440]	validation_0-logloss:0.65392
    [441]	validation_0-logloss:0.65383
    [442]	validation_0-logloss:0.65380
    [443]	validation_0-logloss:0.65379
    [444]	validation_0-logloss:0.65369
    [445]	validation_0-logloss:0.65357
    [446]	validation_0-logloss:0.65349
    [447]	validation_0-logloss:0.65341
    [448]	validation_0-logloss:0.65336
    [449]	validation_0-logloss:0.65336
    [450]	validation_0-logloss:0.65333
    [451]	validation_0-logloss:0.65334
    [452]	validation_0-logloss:0.65329
    [453]	validation_0-logloss:0.65325
    [454]	validation_0-logloss:0.65321
    [455]	validation_0-logloss:0.65319
    [456]	validation_0-logloss:0.65310
    [457]	validation_0-logloss:0.65312
    [458]	validation_0-logloss:0.65311
    [459]	validation_0-logloss:0.65304
    [460]	validation_0-logloss:0.65300
    [461]	validation_0-logloss:0.65299
    [462]	validation_0-logloss:0.65299
    [463]	validation_0-logloss:0.65286
    [464]	validation_0-logloss:0.65287
    [465]	validation_0-logloss:0.65287
    [466]	validation_0-logloss:0.65287
    [467]	validation_0-logloss:0.65276
    [468]	validation_0-logloss:0.65266
    [469]	validation_0-logloss:0.65264
    [470]	validation_0-logloss:0.65259
    [471]	validation_0-logloss:0.65247
    [472]	validation_0-logloss:0.65238
    [473]	validation_0-logloss:0.65233
    [474]	validation_0-logloss:0.65222
    [475]	validation_0-logloss:0.65220
    [476]	validation_0-logloss:0.65219
    [477]	validation_0-logloss:0.65214
    [478]	validation_0-logloss:0.65212
    [479]	validation_0-logloss:0.65211
    [480]	validation_0-logloss:0.65205
    [481]	validation_0-logloss:0.65195
    [482]	validation_0-logloss:0.65195
    [483]	validation_0-logloss:0.65193
    [484]	validation_0-logloss:0.65189
    [485]	validation_0-logloss:0.65188
    [486]	validation_0-logloss:0.65187
    [487]	validation_0-logloss:0.65180
    [488]	validation_0-logloss:0.65172
    [489]	validation_0-logloss:0.65165
    [490]	validation_0-logloss:0.65160
    [491]	validation_0-logloss:0.65156
    [492]	validation_0-logloss:0.65155
    [493]	validation_0-logloss:0.65147
    [494]	validation_0-logloss:0.65141
    [495]	validation_0-logloss:0.65140
    [496]	validation_0-logloss:0.65140
    [497]	validation_0-logloss:0.65139
    [498]	validation_0-logloss:0.65136
    [499]	validation_0-logloss:0.65135
    [500]	validation_0-logloss:0.65134
    [501]	validation_0-logloss:0.65120
    [502]	validation_0-logloss:0.65120
    [503]	validation_0-logloss:0.65108
    [504]	validation_0-logloss:0.65100
    [505]	validation_0-logloss:0.65096
    [506]	validation_0-logloss:0.65092
    [507]	validation_0-logloss:0.65079
    [508]	validation_0-logloss:0.65078
    [509]	validation_0-logloss:0.65068
    [510]	validation_0-logloss:0.65062
    [511]	validation_0-logloss:0.65059
    [512]	validation_0-logloss:0.65052
    [513]	validation_0-logloss:0.65046
    [514]	validation_0-logloss:0.65035
    [515]	validation_0-logloss:0.65033
    [516]	validation_0-logloss:0.65033
    [517]	validation_0-logloss:0.65029
    [518]	validation_0-logloss:0.65026
    [519]	validation_0-logloss:0.65015
    [520]	validation_0-logloss:0.65013
    [521]	validation_0-logloss:0.65008
    [522]	validation_0-logloss:0.65007
    [523]	validation_0-logloss:0.65001
    [524]	validation_0-logloss:0.64997
    [525]	validation_0-logloss:0.64992
    [526]	validation_0-logloss:0.64990
    [527]	validation_0-logloss:0.64988
    [528]	validation_0-logloss:0.64980
    [529]	validation_0-logloss:0.64976
    [530]	validation_0-logloss:0.64973
    [531]	validation_0-logloss:0.64967
    [532]	validation_0-logloss:0.64967
    [533]	validation_0-logloss:0.64966
    [534]	validation_0-logloss:0.64964
    [535]	validation_0-logloss:0.64955
    [536]	validation_0-logloss:0.64954
    [537]	validation_0-logloss:0.64950
    [538]	validation_0-logloss:0.64946
    [539]	validation_0-logloss:0.64945
    [540]	validation_0-logloss:0.64944
    [541]	validation_0-logloss:0.64942
    [542]	validation_0-logloss:0.64941
    [543]	validation_0-logloss:0.64938
    [544]	validation_0-logloss:0.64930
    [545]	validation_0-logloss:0.64919
    [546]	validation_0-logloss:0.64920
    [547]	validation_0-logloss:0.64920
    [548]	validation_0-logloss:0.64919
    [549]	validation_0-logloss:0.64918
    [550]	validation_0-logloss:0.64919
    [551]	validation_0-logloss:0.64919
    [552]	validation_0-logloss:0.64917
    [553]	validation_0-logloss:0.64918
    [554]	validation_0-logloss:0.64917
    [555]	validation_0-logloss:0.64916
    [556]	validation_0-logloss:0.64913
    [557]	validation_0-logloss:0.64914
    [558]	validation_0-logloss:0.64910
    [559]	validation_0-logloss:0.64910
    [560]	validation_0-logloss:0.64911
    [561]	validation_0-logloss:0.64910
    [562]	validation_0-logloss:0.64907
    [563]	validation_0-logloss:0.64902
    [564]	validation_0-logloss:0.64901
    [565]	validation_0-logloss:0.64897
    [566]	validation_0-logloss:0.64893
    [567]	validation_0-logloss:0.64888
    [568]	validation_0-logloss:0.64885
    [569]	validation_0-logloss:0.64881
    [570]	validation_0-logloss:0.64880
    [571]	validation_0-logloss:0.64878
    [572]	validation_0-logloss:0.64878
    [573]	validation_0-logloss:0.64874
    [574]	validation_0-logloss:0.64873
    [575]	validation_0-logloss:0.64871
    [576]	validation_0-logloss:0.64871
    [577]	validation_0-logloss:0.64866
    [578]	validation_0-logloss:0.64855
    [579]	validation_0-logloss:0.64849
    [580]	validation_0-logloss:0.64843
    [581]	validation_0-logloss:0.64842
    [582]	validation_0-logloss:0.64842
    [583]	validation_0-logloss:0.64837
    [584]	validation_0-logloss:0.64835
    [585]	validation_0-logloss:0.64827
    [586]	validation_0-logloss:0.64826
    [587]	validation_0-logloss:0.64815
    [588]	validation_0-logloss:0.64816
    [589]	validation_0-logloss:0.64811
    [590]	validation_0-logloss:0.64812
    [591]	validation_0-logloss:0.64804
    [592]	validation_0-logloss:0.64799
    [593]	validation_0-logloss:0.64798
    [594]	validation_0-logloss:0.64795
    [595]	validation_0-logloss:0.64785
    [596]	validation_0-logloss:0.64785
    [597]	validation_0-logloss:0.64784
    [598]	validation_0-logloss:0.64782
    [599]	validation_0-logloss:0.64782
    [600]	validation_0-logloss:0.64775
    [601]	validation_0-logloss:0.64771
    [602]	validation_0-logloss:0.64761
    [603]	validation_0-logloss:0.64753
    [604]	validation_0-logloss:0.64754
    [605]	validation_0-logloss:0.64750
    [606]	validation_0-logloss:0.64747
    [607]	validation_0-logloss:0.64744
    [608]	validation_0-logloss:0.64744
    [609]	validation_0-logloss:0.64740
    [610]	validation_0-logloss:0.64740
    [611]	validation_0-logloss:0.64733
    [612]	validation_0-logloss:0.64732
    [613]	validation_0-logloss:0.64731
    [614]	validation_0-logloss:0.64731
    [615]	validation_0-logloss:0.64725
    [616]	validation_0-logloss:0.64725
    [617]	validation_0-logloss:0.64721
    [618]	validation_0-logloss:0.64721
    [619]	validation_0-logloss:0.64719
    [620]	validation_0-logloss:0.64716
    [621]	validation_0-logloss:0.64716
    [622]	validation_0-logloss:0.64713
    [623]	validation_0-logloss:0.64713
    [624]	validation_0-logloss:0.64715
    [625]	validation_0-logloss:0.64715
    [626]	validation_0-logloss:0.64712
    [627]	validation_0-logloss:0.64709
    [628]	validation_0-logloss:0.64706
    [629]	validation_0-logloss:0.64706
    [630]	validation_0-logloss:0.64706
    [631]	validation_0-logloss:0.64706
    [632]	validation_0-logloss:0.64705
    [633]	validation_0-logloss:0.64703
    [634]	validation_0-logloss:0.64699
    [635]	validation_0-logloss:0.64688
    [636]	validation_0-logloss:0.64690
    [637]	validation_0-logloss:0.64690
    [638]	validation_0-logloss:0.64692
    [639]	validation_0-logloss:0.64692
    [640]	validation_0-logloss:0.64690
    [641]	validation_0-logloss:0.64689
    [642]	validation_0-logloss:0.64688
    [643]	validation_0-logloss:0.64685
    [644]	validation_0-logloss:0.64683
    [645]	validation_0-logloss:0.64682
    [646]	validation_0-logloss:0.64682
    [647]	validation_0-logloss:0.64682
    [648]	validation_0-logloss:0.64684
    [649]	validation_0-logloss:0.64685
    [650]	validation_0-logloss:0.64682
    [651]	validation_0-logloss:0.64681
    [652]	validation_0-logloss:0.64680
    [653]	validation_0-logloss:0.64678
    [654]	validation_0-logloss:0.64677
    [655]	validation_0-logloss:0.64669
    [656]	validation_0-logloss:0.64667
    [657]	validation_0-logloss:0.64665
    [658]	validation_0-logloss:0.64666
    [659]	validation_0-logloss:0.64663
    [660]	validation_0-logloss:0.64663
    [661]	validation_0-logloss:0.64663
    [662]	validation_0-logloss:0.64663
    [663]	validation_0-logloss:0.64663
    [664]	validation_0-logloss:0.64660
    [665]	validation_0-logloss:0.64654
    [666]	validation_0-logloss:0.64653
    [667]	validation_0-logloss:0.64653
    [668]	validation_0-logloss:0.64653
    [669]	validation_0-logloss:0.64653
    [670]	validation_0-logloss:0.64653
    [671]	validation_0-logloss:0.64653
    [672]	validation_0-logloss:0.64654
    [673]	validation_0-logloss:0.64653
    [674]	validation_0-logloss:0.64653
    [675]	validation_0-logloss:0.64650
    [676]	validation_0-logloss:0.64649
    [677]	validation_0-logloss:0.64645
    [678]	validation_0-logloss:0.64645
    [679]	validation_0-logloss:0.64640
    [680]	validation_0-logloss:0.64638
    [681]	validation_0-logloss:0.64635
    [682]	validation_0-logloss:0.64635
    [683]	validation_0-logloss:0.64634
    [684]	validation_0-logloss:0.64634
    [685]	validation_0-logloss:0.64633
    [686]	validation_0-logloss:0.64631
    [687]	validation_0-logloss:0.64631
    [688]	validation_0-logloss:0.64626
    [689]	validation_0-logloss:0.64626
    [690]	validation_0-logloss:0.64623
    [691]	validation_0-logloss:0.64616
    [692]	validation_0-logloss:0.64612
    [693]	validation_0-logloss:0.64611
    [694]	validation_0-logloss:0.64610
    [695]	validation_0-logloss:0.64609
    [696]	validation_0-logloss:0.64609
    [697]	validation_0-logloss:0.64604
    [698]	validation_0-logloss:0.64605
    [699]	validation_0-logloss:0.64603
    [700]	validation_0-logloss:0.64603
    [701]	validation_0-logloss:0.64603
    [702]	validation_0-logloss:0.64604
    [703]	validation_0-logloss:0.64600
    [704]	validation_0-logloss:0.64600
    [705]	validation_0-logloss:0.64598
    [706]	validation_0-logloss:0.64594
    [707]	validation_0-logloss:0.64594
    [708]	validation_0-logloss:0.64592
    [709]	validation_0-logloss:0.64587
    [710]	validation_0-logloss:0.64584
    [711]	validation_0-logloss:0.64581
    [712]	validation_0-logloss:0.64582
    [713]	validation_0-logloss:0.64584
    [714]	validation_0-logloss:0.64585
    [715]	validation_0-logloss:0.64580
    [716]	validation_0-logloss:0.64576
    [717]	validation_0-logloss:0.64575
    [718]	validation_0-logloss:0.64572
    [719]	validation_0-logloss:0.64571
    [720]	validation_0-logloss:0.64570
    [721]	validation_0-logloss:0.64570
    [722]	validation_0-logloss:0.64569
    [723]	validation_0-logloss:0.64568
    [724]	validation_0-logloss:0.64568
    [725]	validation_0-logloss:0.64567
    [726]	validation_0-logloss:0.64566
    [727]	validation_0-logloss:0.64566
    [728]	validation_0-logloss:0.64566
    [729]	validation_0-logloss:0.64565
    [730]	validation_0-logloss:0.64564
    [731]	validation_0-logloss:0.64562
    [732]	validation_0-logloss:0.64561
    [733]	validation_0-logloss:0.64562
    [734]	validation_0-logloss:0.64561
    [735]	validation_0-logloss:0.64559
    [736]	validation_0-logloss:0.64560
    [737]	validation_0-logloss:0.64559
    [738]	validation_0-logloss:0.64560
    [739]	validation_0-logloss:0.64560
    [740]	validation_0-logloss:0.64560
    [741]	validation_0-logloss:0.64559
    [742]	validation_0-logloss:0.64553
    [743]	validation_0-logloss:0.64553
    [744]	validation_0-logloss:0.64552
    [745]	validation_0-logloss:0.64547
    [746]	validation_0-logloss:0.64547
    [747]	validation_0-logloss:0.64546
    [748]	validation_0-logloss:0.64542
    [749]	validation_0-logloss:0.64539
    [750]	validation_0-logloss:0.64539
    [751]	validation_0-logloss:0.64539
    [752]	validation_0-logloss:0.64538
    [753]	validation_0-logloss:0.64534
    [754]	validation_0-logloss:0.64529
    [755]	validation_0-logloss:0.64524
    [756]	validation_0-logloss:0.64524
    [757]	validation_0-logloss:0.64524
    [758]	validation_0-logloss:0.64525
    [759]	validation_0-logloss:0.64526
    [760]	validation_0-logloss:0.64524
    [761]	validation_0-logloss:0.64523
    [762]	validation_0-logloss:0.64523
    [763]	validation_0-logloss:0.64523
    [764]	validation_0-logloss:0.64522
    [765]	validation_0-logloss:0.64519
    [766]	validation_0-logloss:0.64520
    [767]	validation_0-logloss:0.64515
    [768]	validation_0-logloss:0.64515
    [769]	validation_0-logloss:0.64514
    [770]	validation_0-logloss:0.64515
    [771]	validation_0-logloss:0.64515
    [772]	validation_0-logloss:0.64513
    [773]	validation_0-logloss:0.64513
    [774]	validation_0-logloss:0.64509
    [775]	validation_0-logloss:0.64509
    [776]	validation_0-logloss:0.64509
    [777]	validation_0-logloss:0.64508
    [778]	validation_0-logloss:0.64508
    [779]	validation_0-logloss:0.64508
    [780]	validation_0-logloss:0.64508
    [781]	validation_0-logloss:0.64508
    [782]	validation_0-logloss:0.64508
    [783]	validation_0-logloss:0.64508
    [784]	validation_0-logloss:0.64508
    [785]	validation_0-logloss:0.64505
    [786]	validation_0-logloss:0.64505
    [787]	validation_0-logloss:0.64505
    [788]	validation_0-logloss:0.64505
    [789]	validation_0-logloss:0.64502
    [790]	validation_0-logloss:0.64502
    [791]	validation_0-logloss:0.64499
    [792]	validation_0-logloss:0.64499
    [793]	validation_0-logloss:0.64498
    [794]	validation_0-logloss:0.64498
    [795]	validation_0-logloss:0.64499
    [796]	validation_0-logloss:0.64495
    [797]	validation_0-logloss:0.64495
    [798]	validation_0-logloss:0.64495
    [799]	validation_0-logloss:0.64494
    [800]	validation_0-logloss:0.64494
    [801]	validation_0-logloss:0.64490
    [802]	validation_0-logloss:0.64489
    [803]	validation_0-logloss:0.64487
    [804]	validation_0-logloss:0.64487
    [805]	validation_0-logloss:0.64487
    [806]	validation_0-logloss:0.64486
    [807]	validation_0-logloss:0.64487
    [808]	validation_0-logloss:0.64485
    [809]	validation_0-logloss:0.64485
    [810]	validation_0-logloss:0.64485
    [811]	validation_0-logloss:0.64482
    [812]	validation_0-logloss:0.64482
    [813]	validation_0-logloss:0.64481
    [814]	validation_0-logloss:0.64480
    [815]	validation_0-logloss:0.64480
    [816]	validation_0-logloss:0.64479
    [817]	validation_0-logloss:0.64478
    [818]	validation_0-logloss:0.64477
    [819]	validation_0-logloss:0.64476
    [820]	validation_0-logloss:0.64473
    [821]	validation_0-logloss:0.64473
    [822]	validation_0-logloss:0.64473
    [823]	validation_0-logloss:0.64471
    [824]	validation_0-logloss:0.64468
    [825]	validation_0-logloss:0.64467
    [826]	validation_0-logloss:0.64467
    [827]	validation_0-logloss:0.64467
    [828]	validation_0-logloss:0.64466
    [829]	validation_0-logloss:0.64465
    [830]	validation_0-logloss:0.64465
    [831]	validation_0-logloss:0.64466
    [832]	validation_0-logloss:0.64465
    [833]	validation_0-logloss:0.64464
    [834]	validation_0-logloss:0.64462
    [835]	validation_0-logloss:0.64462
    [836]	validation_0-logloss:0.64462
    [837]	validation_0-logloss:0.64462
    [838]	validation_0-logloss:0.64462
    [839]	validation_0-logloss:0.64462
    [840]	validation_0-logloss:0.64462
    [841]	validation_0-logloss:0.64460
    [842]	validation_0-logloss:0.64457
    [843]	validation_0-logloss:0.64457
    [844]	validation_0-logloss:0.64456
    [845]	validation_0-logloss:0.64456
    [846]	validation_0-logloss:0.64456
    [847]	validation_0-logloss:0.64457
    [848]	validation_0-logloss:0.64456
    [849]	validation_0-logloss:0.64456
    [850]	validation_0-logloss:0.64455
    [851]	validation_0-logloss:0.64451
    [852]	validation_0-logloss:0.64450
    [853]	validation_0-logloss:0.64450
    [854]	validation_0-logloss:0.64448
    [855]	validation_0-logloss:0.64448
    [856]	validation_0-logloss:0.64448
    [857]	validation_0-logloss:0.64448
    [858]	validation_0-logloss:0.64448
    [859]	validation_0-logloss:0.64448
    [860]	validation_0-logloss:0.64447
    [861]	validation_0-logloss:0.64447
    [862]	validation_0-logloss:0.64447
    [863]	validation_0-logloss:0.64447
    [864]	validation_0-logloss:0.64447
    [865]	validation_0-logloss:0.64446
    [866]	validation_0-logloss:0.64446
    [867]	validation_0-logloss:0.64446
    [868]	validation_0-logloss:0.64446
    [869]	validation_0-logloss:0.64445
    [870]	validation_0-logloss:0.64445
    [871]	validation_0-logloss:0.64445
    [872]	validation_0-logloss:0.64447
    [873]	validation_0-logloss:0.64447
    [874]	validation_0-logloss:0.64449
    [875]	validation_0-logloss:0.64449
    [876]	validation_0-logloss:0.64448
    [877]	validation_0-logloss:0.64448
    [878]	validation_0-logloss:0.64450
    [879]	validation_0-logloss:0.64449
    [880]	validation_0-logloss:0.64448
    [881]	validation_0-logloss:0.64448
    [882]	validation_0-logloss:0.64448
    [883]	validation_0-logloss:0.64448
    [884]	validation_0-logloss:0.64448
    [885]	validation_0-logloss:0.64448
    [886]	validation_0-logloss:0.64448
    [887]	validation_0-logloss:0.64447
    [888]	validation_0-logloss:0.64444
    [889]	validation_0-logloss:0.64444
    [890]	validation_0-logloss:0.64444
    [891]	validation_0-logloss:0.64443
    [892]	validation_0-logloss:0.64443
    [893]	validation_0-logloss:0.64441
    [894]	validation_0-logloss:0.64441
    [895]	validation_0-logloss:0.64440
    [896]	validation_0-logloss:0.64441
    [897]	validation_0-logloss:0.64441
    [898]	validation_0-logloss:0.64440
    [899]	validation_0-logloss:0.64441
    [900]	validation_0-logloss:0.64441
    [901]	validation_0-logloss:0.64439
    [902]	validation_0-logloss:0.64438
    [903]	validation_0-logloss:0.64439
    [904]	validation_0-logloss:0.64437
    [905]	validation_0-logloss:0.64431
    [906]	validation_0-logloss:0.64431
    [907]	validation_0-logloss:0.64431
    [908]	validation_0-logloss:0.64431
    [909]	validation_0-logloss:0.64431
    [910]	validation_0-logloss:0.64428
    [911]	validation_0-logloss:0.64428
    [912]	validation_0-logloss:0.64427
    [913]	validation_0-logloss:0.64422
    [914]	validation_0-logloss:0.64422
    [915]	validation_0-logloss:0.64422
    [916]	validation_0-logloss:0.64422
    [917]	validation_0-logloss:0.64422
    [918]	validation_0-logloss:0.64421
    [919]	validation_0-logloss:0.64421
    [920]	validation_0-logloss:0.64421
    [921]	validation_0-logloss:0.64418
    [922]	validation_0-logloss:0.64413
    [923]	validation_0-logloss:0.64412
    [924]	validation_0-logloss:0.64410
    [925]	validation_0-logloss:0.64407
    [926]	validation_0-logloss:0.64407
    [927]	validation_0-logloss:0.64407
    [928]	validation_0-logloss:0.64408
    [929]	validation_0-logloss:0.64408
    [930]	validation_0-logloss:0.64408
    [931]	validation_0-logloss:0.64408
    [932]	validation_0-logloss:0.64405
    [933]	validation_0-logloss:0.64406
    [934]	validation_0-logloss:0.64406
    [935]	validation_0-logloss:0.64404
    [936]	validation_0-logloss:0.64404
    [937]	validation_0-logloss:0.64404
    [938]	validation_0-logloss:0.64404
    [939]	validation_0-logloss:0.64397
    [940]	validation_0-logloss:0.64397
    [941]	validation_0-logloss:0.64397
    [942]	validation_0-logloss:0.64388
    [943]	validation_0-logloss:0.64390
    [944]	validation_0-logloss:0.64390
    [945]	validation_0-logloss:0.64389
    [946]	validation_0-logloss:0.64390
    [947]	validation_0-logloss:0.64390
    [948]	validation_0-logloss:0.64389
    [949]	validation_0-logloss:0.64389
    [950]	validation_0-logloss:0.64389
    [951]	validation_0-logloss:0.64382
    [952]	validation_0-logloss:0.64381
    [953]	validation_0-logloss:0.64381
    [954]	validation_0-logloss:0.64381
    [955]	validation_0-logloss:0.64381
    [956]	validation_0-logloss:0.64381
    [957]	validation_0-logloss:0.64380
    [958]	validation_0-logloss:0.64379
    [959]	validation_0-logloss:0.64379
    [960]	validation_0-logloss:0.64378
    [961]	validation_0-logloss:0.64377
    [962]	validation_0-logloss:0.64375
    [963]	validation_0-logloss:0.64376
    [964]	validation_0-logloss:0.64376
    [965]	validation_0-logloss:0.64374
    [966]	validation_0-logloss:0.64374
    [967]	validation_0-logloss:0.64374
    [968]	validation_0-logloss:0.64374
    [969]	validation_0-logloss:0.64372
    [970]	validation_0-logloss:0.64372
    [971]	validation_0-logloss:0.64372
    [972]	validation_0-logloss:0.64372
    [973]	validation_0-logloss:0.64372
    [974]	validation_0-logloss:0.64371
    [975]	validation_0-logloss:0.64370
    [976]	validation_0-logloss:0.64369
    [977]	validation_0-logloss:0.64368
    [978]	validation_0-logloss:0.64368
    [979]	validation_0-logloss:0.64368
    [980]	validation_0-logloss:0.64367
    [981]	validation_0-logloss:0.64365
    [982]	validation_0-logloss:0.64365
    [983]	validation_0-logloss:0.64364
    [984]	validation_0-logloss:0.64364
    [985]	validation_0-logloss:0.64364
    [986]	validation_0-logloss:0.64360
    [987]	validation_0-logloss:0.64360
    [988]	validation_0-logloss:0.64360
    [989]	validation_0-logloss:0.64359
    [990]	validation_0-logloss:0.64360
    [991]	validation_0-logloss:0.64360
    [992]	validation_0-logloss:0.64359
    [993]	validation_0-logloss:0.64359
    [994]	validation_0-logloss:0.64359
    [995]	validation_0-logloss:0.64358
    [996]	validation_0-logloss:0.64358
    [997]	validation_0-logloss:0.64358
    [998]	validation_0-logloss:0.64358
    [999]	validation_0-logloss:0.64358
    XGBoost Classifier Accuracy: 0.6330306088126472

