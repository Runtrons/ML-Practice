# Dataset Analysis and Processing

**See [this](inference.ipynb) with the stockstats and all in one. If you want to use the 22 year data (Isnt the cleanest data) drop the volume column:
df.drop('volume', axis=1, inplace=True). The tesla dataset is the most trustworthy dataset coming from a reliable source.

^^^^^^^^^^^^^^^^^^^

## Data Source

I downloaded the dataset from [this Russian financial website](https://www.finam.ru/quote/forex/eurusd/export/). The website is in Russian, but I used my browser's translate feature. 

### Key Points about the Dataset
- **Verification**: The data was briefly verified for accuracy.
- **Time Zone**: The data is in Moscow time.

## Accessing the Dataset

You can download the dataset I used at the beginning of this process from my Google Drive, as the files are too large for GitHub:
- [Main Google Drive Folder](https://drive.google.com/drive/folders/1NIw2Kqc043LLIK7ZXrx8wYRol7zuFjP2?usp=drive_link)
- [Specific Dataset File](https://drive.google.com/file/d/1HAWgJpsows16hIDv3vg1szH9XQ7CGR6C/view?usp=drive_link)

All of the datasets are in the main google drive folder.

## Data Processing

After downloading the data, the following steps were undertaken:
1. **Time Zone Conversion**: Converted the time zone from Moscow time.
2. **Data Cleaning and Preparation**: Cleaned and prepared the data for analysis.
3. **Indicator Addition**: Added various indicators. (Note: If there are errors, they likely occurred here.)

The processing script is documented in the markdown file "dataset" which goes over how I made the dataset. ([(Here)](dataset.md)):

**Author's Note**: The dataset and scripts were generated with the assistance of ChatGPT, under my guidance as the prompt master.
**Author's Note2**: So was the read me as you can definitly tell :).

## Final Dataset

The best-performing dataset for use can be found here:
- [Optimized Dataset File](https://drive.google.com/file/d/1TbTsfjtshYY6l_2K6uCzVom-ZqGqV4vh/view?usp=drive_link)

I have created 60 variations of the dataset, some of which are in the Google Drive folder. The head of the final file is shown below:

```python
OPEN,HIGH,LOW,CLOSE,VOL,SMA_50,EMA_50,RSI_14,MACD,MACD_Signal,SMA_20,Bollinger_Upper,Bollinger_Lower,time,day_of_week,%K,%D,ATR,Target
1.0402,1.04036,1.03992,1.04034,1039,1.04048,1.04042,39.51613,-9e-05,-9e-05,1.04037,1.04082,1.03993,22:21,2,63.09523809524188,53.96825396825272,0.0003749999999999,1
1.04036,1.04037,1.04004,1.04017,933,1.04048,1.04041,34.75177,-9e-05,-9e-05,1.04035,1.04077,1.03992,22:22,2,45.56962025317808,53.681936909787,0.0003799999999999,1
```

The target in the dataset aims to predict market movement: "1" indicates an upward movement in the specified time period, while "0" indicates a downward trend.

## Experimentation and Results

To determine the best time period for predictions, I ran the following script:

**Main Script** ([main_script.py](main_script.py)):
```python
import os
from multiprocessing import Pool
from processing_functions import process_file  # Import the function

def main():
    folder_path = '/Users/ronangrant/Vault/indicators_and_times'
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    files.sort()

    with Pool(5) as pool:
        results = pool.map(process_file, files)

    for result in results:
        print(result)

if __name__ == '__main__':
    main()

```

This script utilizes functions from another script:

**Processing Functions** ([processing_functions.py](processing_functions.py)):
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

def process_file(file):
    print(f"Starting processing on {os.path.basename(file)}")

    df = pd.read_csv(file)

    # Assuming the 'time' column is in 'HH:MM' format
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
    df['minute'] = pd.to_datetime(df['time'], format='%H:%M').dt.minute

    df.drop('time', axis=1, inplace=True)

    X = df.drop('Target', axis=1)
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Running Random Forest Classifier on {os.path.basename(file)} - Accuracy: {accuracy}")
    return (os.path.basename(file), accuracy)

```

The scripts attempt to process all 60 datasets with basic forest tree settings and output the results. It uses multithreading so it does not take forever. These datasets were generated using [this script](main_script.py).

### Results

My results for the datasets in Google Drive are as follows:
- **60min**: 0.927 accuracy
- **30min**: 0.892 accuracy
- **5min**: 0.710 accuracy

You can reproduce these results using [inference.py](inference.py):
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump

def train_and_save_model(file_path, model_save_path):
    print(f"Starting processing on {file_path}")

    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'time' from string to numerical values
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
    df['minute'] = pd.to_datetime(df['time'], format='%H:%M').dt.minute
    df.drop('time', axis=1, inplace=True)

    # Define features and target
    X = df.drop('Target', axis=1)
    y = df['Target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for 35min dataset: {accuracy}")

    # Save the trained model
    dump(rf_classifier, model_save_path)
    print(f"Model saved to {model_save_path}")

# File paths
dataset_path = '/Users/ronangrant/Vault/indicators_and_times/dataset_60min.csv'
model_save_path = '/Users/ronangrant/Vault/trading_models/35min_rf_model.joblib'

# Train the model and save it
train_and_save_model(dataset_path, model_save_path)

```

If you want the exact model I used to get the 92% results you can get it here too: https://drive.google.com/file/d/19aqpTFvfxXdEhSaBCFkLiHWbFGkF6dQI/view?usp=drive_link

I obviously do not bealive these reflect the actual results, thats why I want you to look over it and see if you can figure out what it is wrong and guide me to the right direction.
