from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model, encoder, and unique values
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('unique_values.pkl', 'rb') as f:
    unique_values = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    city_encoded = encoder.transform([[data['city']]]).toarray()
    features = np.array([data['house_age'], data['rooms'], data['bedrooms']])
    features = np.concatenate([features, city_encoded[0]])
    prediction = model.predict([features])
    return jsonify({'price': prediction[0]})

@app.route('/unique_values', methods=['GET'])
def get_unique_values():
    return jsonify(unique_values)

if __name__ == '__main__':
    app.run(debug=True)