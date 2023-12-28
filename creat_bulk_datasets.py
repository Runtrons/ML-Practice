import pandas as pd

def create_target_variable(df, future_minutes):
    """
    Creates a target variable indicating if the price goes up or down in the specified future timeframe.
    1 indicates an increase, and 0 indicates a decrease or no change.
    """
    # Shift the closing price by the specified future minutes
    df['Future_Close'] = df['CLOSE'].shift(-future_minutes)

    # Determine if the future closing price is higher than the current closing price
    df['Target'] = (df['Future_Close'] > df['CLOSE']).astype(int)

    # Drop the last 'future_minutes' rows as they won't have a future price
    df.dropna(subset=['Future_Close'], inplace=True)

    # Optionally, drop the 'Future_Close' column if it's no longer needed
    df.drop(columns=['Future_Close'], inplace=True)

    return df

# Load your dataset
file_path = '/Users/ronangrant/Vault/training_data/updated_data_with_new_indicators.csv'
df = pd.read_csv(file_path)

# Drop existing 'Target' column if it exists and remove rows with missing values
df.drop(columns='Target', errors='ignore', inplace=True)
df.dropna(inplace=True)

# Generate and save datasets for each time frame (1 to 60 minutes)
for minutes in range(1, 61):
    df_modified = create_target_variable(df, future_minutes=minutes)
    output_file = f'/Users/ronangrant/Vault/indicators_and_times/modified_dataset_{minutes}min.csv'
    df_modified.to_csv(output_file, index=False)

