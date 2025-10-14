# 📊 EDA & LSTM Prediction Dashboard

Automated Exploratory Data Analysis and 5-Year Forecasting using Multi-Output LSTM Model for Natural Capital Accounting (NCA) intensity predictions.

## 🌟 Features

### 1. **Automatic EDA Generation**
Upload a dataset to automatically generate comprehensive exploratory data analysis:
- Dataset overview and statistics
- Missing values analysis  
- Correlation heatmap
- Distribution plots for numerical features
- Box plots for outlier detection
- Categorical variable analysis

### 2. **LSTM Time Series Forecasting**
Predict **intensity_nca** (NCA/GDP ratio) for the **next 5 years**:
- **Model Type**: Multi-Output LSTM (TensorFlow/Keras)
- **Lookback Period**: 5 consecutive years
- **Forecast Horizon**: 5 future years
- **Input Method**: Manual data entry form
- **Features**: 7 economic indicators
- **Visualization**: Interactive charts showing historical + forecast
- **Download**: CSV results with predictions

## 📋 Required Data for Forecasting

### Input Requirements
Enter data for **5 consecutive years** with these 7 features:

| Feature | Description |
|---------|-------------|
| `nca` | Natural Capital Accounting value |
| `gdp` | Gross Domestic Product |
| `growth_gdp` | GDP growth rate |
| `growth_nca` | NCA growth rate |
| `loggrowth_gdp` | Log of GDP growth |
| `loggrowth_nca` | Log of NCA growth |
| `intensity_nca` | NCA/GDP ratio (auto-calculated if not provided) |

### Optional Information
- `year` - For temporal context
- `code` - Country/entity code (e.g., USA, IDN, CHN)
- `countryname` - Full country/entity name

## 🚀 Installation

### 1. Create Virtual Environment
```bash
python -m venv .venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Running the Application

### Option 1: Streamlit Interface (Recommended)

Run the Streamlit web interface:

```bash
streamlit run app_streamlit.py
```

The application will be available at `http://localhost:8501`

**Features:**
- 📝 Manual data entry form for 5 years
- 📊 Automatic EDA generation with visualizations  
- 🔮 5-year forecast with interactive charts
- 📥 Downloadable CSV results
- 🎨 Beautiful, responsive UI

### Option 2: FastAPI Backend

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8005
```

The API will be available at `http://127.0.0.1:8005`

Interactive API documentation (Swagger UI): `http://127.0.0.1:8005/docs`

## 📖 Usage Guide

### Using Streamlit Interface (Easiest)

1. **Run the app:**
   ```bash
   streamlit run app_streamlit.py
   ```

2. **Navigate to EDA Tab:**
   - Upload CSV/Excel file
   - View automatic analysis and visualizations

3. **Navigate to Predictions Tab:**
   - Enter country information (optional)
   - Fill in economic data for 5 consecutive years:
     - Year, NCA, GDP, Growth rates, Log growth rates
   - Click "🔮 Generate 5-Year Forecast"
   - View predictions and visualization
   - Download results as CSV

### Example Input Data

| Year | NCA | GDP | Growth GDP | Growth NCA | Log Growth GDP | Log Growth NCA |
|------|-----|-----|------------|------------|----------------|----------------|
| 2020 | 100 | 20000 | 0.03 | 0.02 | 0.029 | 0.019 |
| 2021 | 105 | 21000 | 0.05 | 0.05 | 0.048 | 0.048 |
| 2022 | 110 | 22000 | 0.048 | 0.048 | 0.046 | 0.046 |
| 2023 | 115 | 23000 | 0.045 | 0.045 | 0.044 | 0.044 |
| 2024 | 120 | 24000 | 0.043 | 0.043 | 0.042 | 0.042 |

**Output:** Predictions for years 2025-2029

## 🔌 API Endpoints

### 1. Upload & Generate EDA
**POST** `/upload-eda`

Upload a CSV or Excel file to automatically generate exploratory data analysis.

### 2. Make LSTM Predictions  
**POST** `/predict`

Generate 5-year forecasts using the LSTM model.

**Request:**
- File: CSV or Excel with 5+ rows of historical data

**Response:**
```json
{
  "session_id": "uuid",
  "n_sequences": 1,
  "predictions_shape": [1, 5],
  "lookback": 5,
  "horizon": 5,
  "target": "intensity_nca",
  "predictions_summary": {
    "min": 0.005,
    "max": 0.006,
    "mean": 0.0055
  },
  "download_url": "/download/uuid_predictions.csv"
}
```

### 3. Get Model Information
**GET** `/model-info`

Retrieve LSTM model information.

### 4. Get Model Features
**GET** `/model-features`

Get list of required features for predictions.

### 5. Download Results
**GET** `/download/{filename}`

Download prediction results as CSV.

### 6. Get Plots
**GET** `/plots/{filename}`

Retrieve generated EDA visualization images.

### 7. Health Check
**GET** `/health`

Check API health status.

## 🐍 Python API Example

```python
import requests
import pandas as pd

# 1. Upload dataset for EDA
with open('dataset.csv', 'rb') as f:
    response = requests.post('http://127.0.0.1:8005/upload-eda', 
                           files={'file': f})
    eda_result = response.json()
    print(f"Dataset: {eda_result['dataset_info']['n_rows']} rows")

# 2. Make 5-year forecast
with open('historical_data.csv', 'rb') as f:
    response = requests.post('http://127.0.0.1:8005/predict',
                           files={'file': f})
    result = response.json()
    print(f"Generated {result['n_sequences']} forecast sequences")
    print(f"Predictions for next {result['horizon']} years")
    
    # Download results
    csv_response = requests.get(f"http://127.0.0.1:8005{result['download_url']}")
    with open('forecasts.csv', 'wb') as f:
        f.write(csv_response.content)

# 3. Get model information
response = requests.get('http://127.0.0.1:8005/model-info')
model_info = response.json()
print(f"Model: {model_info['model_name']}")
print(f"Lookback: {model_info['lookback']} years")
print(f"Horizon: {model_info['horizon']} years")
```

## 📁 Project Structure

```plaintext
nca-app/
├── .venv/                    # Virtual environment
├── model/                    # LSTM model files
│   ├── multi_output_lstm_h5_lookback5.keras  # Keras model
│   ├── scaler_multi.pkl      # StandardScaler for features
│   └── app.py                # Flask app reference
├── plots/                    # Generated EDA visualizations
├── results/                  # Prediction outputs (CSV)
├── uploads/                  # Temporary file uploads
├── static/                   # Static files (if any)
├── app_streamlit.py          # ✅ Main Streamlit web interface
├── main.py                   # ✅ FastAPI backend server
├── eda_module.py             # ✅ EDA generation module
├── prediction_lstm.py        # ✅ LSTM prediction module
├── models.py                 # ✅ Pydantic data models
├── requirements.txt          # ✅ Python dependencies
├── Dockerfile                # Docker configuration
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # ✅ This file
```

## 🤖 Model Information

### Multi-Output LSTM Model
- **Architecture**: LSTM (Long Short-Term Memory)
- **Framework**: TensorFlow/Keras
- **Input Shape**: (batch_size, 5 timesteps, 7 features)
- **Output Shape**: (batch_size, 5 predictions)
- **Lookback Period**: 5 years
- **Forecast Horizon**: 5 years  
- **Target Variable**: intensity_nca (NCA/GDP ratio)
- **Preprocessing**: StandardScaler fitted on training data

### Model Files
- **Model**: `model/multi_output_lstm_h5_lookback5.keras` (444 KB)
- **Scaler**: `model/scaler_multi.pkl` (790 bytes)

## 📦 Dependencies

Key libraries:
- **FastAPI** - Web API framework
- **Streamlit** - Interactive web interface
- **TensorFlow** - LSTM model inference
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualizations
- **Scikit-learn** - Data scaling

## 📝 Notes

- **Supported Formats**: CSV, Excel (.xlsx, .xls) for EDA
- **Manual Entry**: For predictions, use the Streamlit form
- **Minimum Data**: 5 consecutive years required
- **Automatic Scaling**: Features are scaled using pre-fitted StandardScaler
- **Intensity Calculation**: If not provided, intensity_nca = nca / gdp
- **Output**: Predictions are returned as intensity_nca values

## 🐳 Docker Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker deployment instructions.

## 📄 License

This project is for educational purposes as part of Data Analysis coursework.

## 👥 Contributors

- Developed for Natural Capital Accounting analysis
- Semester 7 - Data Analyst Course
