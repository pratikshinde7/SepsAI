import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
# New imports for the Deep Learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Data Loading and Preprocessing (Unchanged) ---
def load_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return df

def preprocess_data(df):
    print("\nPreprocessing data...")
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    print("Missing values handled.")
    return df

# --- XGBoost Feature Engineering (Unchanged) ---
def engineer_features_for_xgboost(df):
    print("\nEngineering features for XGBoost...")
    grouped = df.groupby('PatientID')
    features_df = grouped.agg(['mean', 'std', 'max', 'min']).reset_index()
    features_df.columns = ['_'.join(col).strip() for col in features_df.columns.values]
    features_df.rename(columns={'PatientID_': 'PatientID'}, inplace=True)
    labels_df = df.groupby('PatientID').last()['SepsisLabel'].reset_index()
    final_df = pd.merge(features_df, labels_df, on='PatientID')
    print("XGBoost features complete.")
    return final_df

# --- XGBoost Model Training (Unchanged) ---
def train_xgboost_model(df):
    print("\n--- Training XGBoost Model ---")
    y = df['SepsisLabel']
    X = df.drop(['PatientID', 'SepsisLabel'], axis=1)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nXGBoost Model Accuracy: {accuracy * 100:.2f}%")
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return model, scaler

# --- NEW: Data Preparation for LSTM ---
def prepare_data_for_lstm(df):
    """
    Prepares the data in a sequence format that the LSTM model can understand.
    Each patient will be a "sample", with 6 "timesteps" of their vital signs.
    """
    print("\nPreparing data for LSTM...")
    # Get a list of all unique patient IDs
    patient_ids = df['PatientID'].unique()
    
    # We will create sequences of data for each patient
    sequences = []
    labels = []
    
    # Define the features we will use for the sequence
    features_to_use = ['HR', 'Temp', 'SBP', 'Resp', 'O2Sat']
    
    for pid in patient_ids:
        patient_data = df[df['PatientID'] == pid][features_to_use]
        sequences.append(patient_data.values)
        labels.append(df[df['PatientID'] == pid]['SepsisLabel'].iloc[-1]) # Final label
        
    # Pad sequences to ensure they are all the same length (important for LSTM)
    # Our sample data is already uniform, but this is good practice
    X = pad_sequences(sequences, padding='post', dtype='float32')
    y = np.array(labels)
    
    print("LSTM data preparation complete.")
    return X, y

# --- NEW: LSTM Model Training ---
def train_lstm_model(X, y):
    """Builds, trains, and evaluates the LSTM deep learning model."""
    print("\n--- Training LSTM Deep Learning Model ---")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build the LSTM model architecture
    model = Sequential([
        # The LSTM layer has a "memory" to process sequences
        LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2), # Helps prevent overfitting
        # The Dense layer is a standard fully connected neural network layer
        Dense(units=1, activation='sigmoid') # Sigmoid is used for binary (0 or 1) classification
    ])
    
    # Compile the model, defining the optimizer and loss function
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Train the model on our data
    print("\nTraining LSTM...")
    model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test), verbose=0)
    
    # Evaluate the model's performance on the test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nLSTM Model Accuracy: {accuracy * 100:.2f}%")

    # Get a full classification report
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype("int32") # Convert probabilities to 0 or 1
    print("\nLSTM Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    data_filepath = 'sample_patient_data.csv'
    
    # --- Run the full pipeline ---
    patient_data = load_data(data_filepath)
    patient_data_processed = preprocess_data(patient_data)
    
    # --- XGBoost Pipeline ---
    patient_features_xgb = engineer_features_for_xgboost(patient_data_processed)
    xgb_model, xgb_scaler = train_xgboost_model(patient_features_xgb)

    # --- LSTM Pipeline ---
    X_lstm, y_lstm = prepare_data_for_lstm(patient_data_processed)
    lstm_model = train_lstm_model(X_lstm, y_lstm)

