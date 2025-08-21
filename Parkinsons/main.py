# ======================================================
# 1. Importing Necessary Libraries
# ======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import warnings
import time

warnings.filterwarnings('ignore')

# ======================================================
# 2. Data Loading and Exploration
# ======================================================
def load_data(filepath):
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the dataset CSV file.

    Returns:
        DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return None

def explore_data(data):
    """
    Explore the dataset.

    Args:
        data (DataFrame): Input dataset.

    Prints:
        - First and last rows, shape, info, null counts, statistics, and target distribution.
    """
    print("\nFirst 5 rows of the dataset:\n", data.head())
    print("\nLast 5 rows of the dataset:\n", data.tail())
    print("\nDataset shape:", data.shape)
    print("\nDataset info:\n")
    data.info()
    print("\nNull values in each column:\n", data.isnull().sum())
    print("\nStatistical summary:\n", data.describe())
    print("\nTarget variable distribution:\n", data['status'].value_counts())

# ======================================================
# 3. Data Preprocessing
# ======================================================
def preprocess_data(data):
    """
    Preprocess the dataset by separating features and target, and splitting into train-test sets.

    Args:
        data (DataFrame): Input dataset.

    Returns:
        tuple: X_train, X_test, Y_train, Y_test
    """
    # Separating features and target
    X = data.drop(columns=['name', 'status'], axis=1)
    Y = data['status']
    # Splitting into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    return X_train, X_test, Y_train, Y_test

# ======================================================
# 4. Model Training
# ======================================================
def train_model(X_train, Y_train, kernel='poly'):
    """
    Train an SVM model.

    Args:
        X_train (DataFrame): Training features.
        Y_train (Series): Training target.
        kernel (str): Kernel type for the SVM.

    Returns:
        model: Trained SVM model.
        float: Training time.
    """
    start_time = time.time()
    model = svm.SVC(kernel=kernel)
    model.fit(X_train, Y_train)
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

# ======================================================
# 5. Model Evaluation
# ======================================================
def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained SVM model.
        X_test (DataFrame): Test features.
        Y_test (Series): Test target.

    Returns:
        float: Accuracy score on test data.
    """
    predictions = model.predict(X_test)
    return accuracy_score(Y_test, predictions)

# ======================================================
# 6. Predictive System
# ======================================================
def make_prediction(model, input_data):
    """
    Predict for a single data point.

    Args:
        model: Trained model.
        input_data (tuple): Input data.

    Prints:
        - Prediction result.
    """
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0] == 0:
        print("\nPrediction: The person does not have Parkinson's disease.")
    else:
        print("\nPrediction: The person has Parkinson's disease.")

# ======================================================
# 7. Save and Load Model
# ======================================================
def save_model(model, filename='parkinsons_model.sav'):
    """
    Save the trained model to a file.

    Args:
        model: Trained SVM model.
        filename (str): File name for saving the model.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename='parkinsons_model.sav'):
    """
    Load a saved model from a file.

    Args:
        filename (str): File name of the saved model.

    Returns:
        model: Loaded model.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

# ======================================================
# 8. Main Execution
# ======================================================
if __name__ == "__main__":
    filepath = 'parkinsons.csv'
    
    # Step 1: Load the data
    data = load_data(filepath)
    if data is not None:
        # Step 2: Explore the data
        explore_data(data)

        # Step 3: Preprocess the data
        X_train, X_test, Y_train, Y_test = preprocess_data(data)
        
        # Step 4: Train the model
        model, training_time = train_model(X_train, Y_train)
        print("\nTraining Time:", training_time, "seconds")

        # Step 5: Evaluate the model
        accuracy = evaluate_model(model, X_test, Y_test)
        print("\nTest Data Accuracy:", accuracy)

        # Step 6: Make a prediction
        sample_input = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 
                        0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.77500, 
                        0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)
        make_prediction(model, sample_input)

        # Step 7: Save the model
        save_model(model)
        print("\nModel saved successfully.")

        # Step 8: Verify by loading the model
        loaded_model = load_model()
        print("\nLoaded model verified successfully.")
