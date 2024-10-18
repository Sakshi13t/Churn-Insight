from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('CP_model.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = request.form.to_dict()

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert necessary columns to appropriate types
    input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(int)
    input_df['MonthlyCharges'] = input_df['MonthlyCharges'].astype(float)
    input_df['TotalCharges'] = input_df['TotalCharges'].astype(float)

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df, columns=[
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ])

    # Ensure the columns match the training data
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make predictions using the model
    prediction = model.predict(input_df)

    # Convert prediction to standard Python types
    prediction_result = prediction[0].item()  # This converts a single value array to a standard Python type

    # Create human-readable output
    if prediction_result == 1:
        prediction_message = "Customer is likely to churn."
    else:
        prediction_message = "Customer is likely to stay."

    # Return the prediction result
    return jsonify({'prediction': prediction_result, 'message': prediction_message})



if __name__ == '__main__':
    app.run(debug=True)
