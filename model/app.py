
from flask import Flask, request, render_template_string, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

# Load model and scaler
model = tf.keras.models.load_model('models_deployment/multi_output_lstm_h5_lookback5.keras')
with open('models_deployment/scaler_multi.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string(HTML_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        country = request.form['country']

        # Extract 5 years of data
        years = []
        data_matrix = []

        for i in range(1, 6):  # Years 1-5
            year = int(request.form[f'year{i}'])
            nca = float(request.form[f'nca{i}'])
            gdp = float(request.form[f'gdp{i}'])
            growth_gdp = float(request.form[f'growth_gdp{i}'])
            growth_nca = float(request.form[f'growth_nca{i}'])
            loggrowth_gdp = float(request.form[f'loggrowth_gdp{i}'])
            loggrowth_nca = float(request.form[f'loggrowth_nca{i}'])
            intensity = float(request.form[f'intensity{i}'])

            years.append(year)
            data_matrix.append([nca, gdp, growth_gdp, growth_nca, loggrowth_gdp, loggrowth_nca, intensity])

        # Prepare data for prediction
        df_input = pd.DataFrame(data_matrix, columns=[
            'nca', 'gdp', 'growth_gdp', 'growth_nca', 
            'loggrowth_gdp', 'loggrowth_nca', 'intensity_nca'
        ])

        # Scale data
        scaled_data = scaler.transform(df_input)
        input_sequence = scaled_data.reshape(1, 5, 7)

        # Make prediction
        predictions = model.predict(input_sequence)
        pred_values = predictions[0].tolist()

        # Generate future years
        future_years = [max(years) + i + 1 for i in range(5)]

        result = {
            'country': country,
            'input_years': years,
            'predicted_years': future_years,
            'predictions': pred_values,
            'success': True
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
