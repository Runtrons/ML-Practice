# Dataset Analysis and Processing

## Data Source

I downloaded the dataset from [this Russian financial website](https://www.finam.ru/quote/forex/eurusd/export/). The website is in Russian, but I used my browser's translate feature. 

### Key Points about the Dataset
- **Verification**: The data was briefly verified for accuracy.
- **Time Zone**: The data is in Moscow time.

## Accessing the Dataset

You can download the dataset I used at the beginning of this process from my Google Drive, as the files are too large for GitHub:
- [Main Google Drive Folder](https://drive.google.com/drive/folders/1NIw2Kqc043LLIK7ZXrx8wYRol7zuFjP2?usp=drive_link)
- [Specific Dataset File](https://drive.google.com/file/d/1HAWgJpsows16hIDv3vg1szH9XQ7CGR6C/view?usp=drive_link)

## Data Processing

After downloading the data, the following steps were undertaken:
1. **Time Zone Conversion**: Converted the time zone from Moscow time.
2. **Data Cleaning and Preparation**: Cleaned and prepared the data for analysis.
3. **Indicator Addition**: Added various indicators. (Note: If there are errors, they likely occurred here.)

The processing script is documented in the markdown file "dataset".

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

**Main Script** ([main_script.py](link-to-main-script)):
```python
# Placeholder for code1
```

This script utilizes functions from another script:

**Processing Functions** ([processing_functions.py](link-to-processing-functions)):
```python
# Placeholder for code2
```

The scripts attempt to process all 60 datasets with basic forest tree settings and output the results. These datasets were generated using [this script](link-to-code).

### Results

My results for the datasets in Google Drive are as follows:
- **60min**: 0.927 accuracy
- **30min**: 0.892 accuracy
- **5min**: 0.710 accuracy

You can reproduce these results using [inference.py](link-to-inference-script).
