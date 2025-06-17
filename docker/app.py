import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Load the pre-trained model
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Convert input to DataFrame for prediction
        input_data = data["input"]
        
        # Create a DataFrame from the input data
        df = pd.DataFrame([input_data])
        
        # Process the data similarly to how we trained the model
        
        # Handle categorical features
        if 'Sex' in df.columns:
            # Create dummy variable for Sex (1 for male, 0 for female)
            df['Sex_male'] = (df['Sex'] == 'male').astype(int)
            df = df.drop('Sex', axis=1)
        else:
            df['Sex_male'] = 0  # Default
            
        # Handle Embarked categorical feature
        if 'Embarked' in df.columns:
            df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
            df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
            df = df.drop('Embarked', axis=1)
        else:
            df['Embarked_Q'] = 0  # Default
            df['Embarked_S'] = 0  # Default
            
        # Drop columns not used in the model
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin'] 
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
                
        # Expected columns based on training
        expected_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
        
        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # Default value if column is missing
        
        # Ensure columns are in the correct order
        df = df[expected_columns]
        
        # Make prediction
        predictions = model.predict(df)
        prediction_proba = model.predict_proba(df)[0].tolist() if hasattr(model, 'predict_proba') else None
        
        # Return prediction result (0 = did not survive, 1 = survived)
        result = {
            "survived": bool(predictions[0]),
            "survived_value": int(predictions[0]),
            "message": "Passenger would likely survive" if predictions[0] else "Passenger would likely not survive"
        }
        
        # Add probability if available
        if prediction_proba:
            result["probability"] = {
                "not_survived": prediction_proba[0],
                "survived": prediction_proba[1]
            }
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)