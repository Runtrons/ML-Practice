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
