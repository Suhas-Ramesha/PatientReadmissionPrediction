from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model from the correct path
model = joblib.load('./data/diabetes_readmission_model.pkl')  # Update this path

# Root route
@app.route('/')
def home():
    return "Welcome to the Diabetes Readmission Prediction API! Use the /predict endpoint to make predictions."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json
    
    # Convert JSON data to a DataFrame
    input_df = pd.DataFrame([data])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_df)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)