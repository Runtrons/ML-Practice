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
