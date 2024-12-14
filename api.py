from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('house_price_predictor.pkl')

DEFAULT_VALUES = {
    'CRIM': 0.0, 'ZN': 0.0, 'INDUS': 0.0, 'CHAS': 0.0, 'NOX': 0.0, 
    'RM': 5.0, 'AGE': 50.0, 'DIS': 1.0, 'RAD': 1, 'TAX': 300.0, 
    'PTRATIO': 15.0, 'B': 300.0, 'LSTAT': 10.0
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        features = np.array(
            [
                data.get(param, DEFAULT_VALUES[param]) for  param in DEFAULT_VALUES
            ]
        ).reshape(1, -1)
        
        prediction = model.predict(features)
        
        return jsonify({
            'prediction': prediction.tolist()[0][0],
            'message': f'Une telle maison est estimée à {prediction[0][0]*1000}$',
            'used_values': {param: data.get(param, DEFAULT_VALUES[param]) for param in DEFAULT_VALUES}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
