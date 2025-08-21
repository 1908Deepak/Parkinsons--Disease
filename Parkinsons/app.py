# ======================================================
# 1. Import Necessary Libraries
# ======================================================
import numpy as np
import pickle
import streamlit as st

# ======================================================
# 2. Utility Functions
# ======================================================

def load_model(model_path='parkinsons_model.sav'):
    """
    Load the saved machine learning model from the specified file path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: The loaded machine learning model.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Ensure the model file path is correct.")
        return None


def predict_disease(input_data, model):
    """
    Predict whether a person has Parkinson's disease based on input data.

    Args:
        input_data (list): List of input features for the prediction.
        model: The trained machine learning model.

    Returns:
        str: Prediction result as a string.
    """
    try:
        input_data_as_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)

        if prediction[0] == 0:
            return "The person does not have Parkinson's disease."
        else:
            return "The person has Parkinson's disease."
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error in prediction."

# ======================================================
# 3. Input Validation Function
# ======================================================

def validate_inputs(inputs):
    """
    Validate user inputs to ensure all fields are filled and contain numeric values.

    Args:
        inputs (list): List of input values provided by the user.

    Returns:
        tuple: (bool, str) - Validation status and error message (if any).
    """
    try:
        # Check if any input is empty
        if not all(inputs):
            return False, "All fields are required for prediction."
        
        # Convert inputs to float
        inputs = list(map(float, inputs))
        return True, inputs
    except ValueError:
        return False, "Please enter valid numeric values for all fields."

# ======================================================
# 4. Streamlit Application
# ======================================================

def main():
    """
    Main function to define the structure of the Streamlit application.
    """
    # --------------------------------------------------
    # App Title and Description
    # --------------------------------------------------
    st.title("Parkinson's Disease Prediction App")
    #------------------------
    # st.markdown(
    #     """
    #     This application uses a machine learning model to predict whether a person has Parkinson's disease.
        
    #     **Features for Prediction**:
    #     - MDVP: Fo (Hz)
    #     - MDVP: Fhi (Hz)
    #     - MDVP: Flo (Hz)
    #     - MDVP: Jitter (%)
    #     - MDVP: Jitter (Abs)
    #     - MDVP: RAP
    #     - MDVP: PPQ
    #     - Jitter: DDP
    #     - MDVP: Shimmer
    #     - MDVP: Shimmer (dB)
    #     - Shimmer: APQ3
    #     - Shimmer: APQ5
    #     - MDVP: APQ
    #     - Shimmer: DDA
    #     - NHR
    #     - HNR
    #     - RPDE
    #     - DFA
    #     - Spread1
    #     - Spread2
    #     - D2
    #     - PPE
    #     """
    # )
    #---------------

    # --------------------------------------------------
    # Input Fields for User Data
    # --------------------------------------------------
    st.header("Enter Patient's Health Parameters")
    features = [
        "MDVP: Fo (Hz)", "MDVP: Fhi (Hz)", "MDVP: Flo (Hz)", "MDVP: Jitter (%)",
        "MDVP: Jitter (Abs)", "MDVP: RAP", "MDVP: PPQ", "Jitter: DDP",
        "MDVP: Shimmer", "MDVP: Shimmer (dB)", "Shimmer: APQ3", "Shimmer: APQ5",
        "MDVP: APQ", "Shimmer: DDA", "NHR", "HNR", "RPDE", "DFA",
        "Spread1", "Spread2", "D2", "PPE"
    ]

    inputs = [st.text_input(f"{feature}") for feature in features]

    # --------------------------------------------------
    # Prediction Button and Result Display
    # --------------------------------------------------
    st.header("Prediction Result")
    if st.button("Get Test Result"):
        # Validate inputs
        is_valid, validated_data_or_error = validate_inputs(inputs)

        if is_valid:
            # Make prediction
            result = predict_disease(validated_data_or_error, loaded_model)
            st.success(result)
        else:
            # Display validation error
            st.error(validated_data_or_error)

# ======================================================
# 5. Application Entry Point
# ======================================================

if __name__ == '__main__':
    # Load the model at the start of the application
    loaded_model = load_model('parkinsons_model.sav')

    # Run the application if the model is loaded successfully
    if loaded_model:
        main()
