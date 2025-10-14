# EDA & Prediction API with FastAPI

Automated Exploratory Data Analysis and XGBoost Prediction Pipeline using FastAPI.

## Features

- **Automatic EDA Generation**: Upload a dataset and automatically generate comprehensive exploratory data analysis including:
  - Dataset overview and statistics
  - Missing values analysis
  - Correlation heatmap
  - Distribution plots
  - Box plots for outlier detection
  - Categorical variable analysis

- **XGBoost Predictions with Auto Feature Engineering**: Use pre-trained XGBoost model to make predictions
  - **Automatic Feature Engineering**: Automatically creates lag features, rolling means, and intensity features from raw data
  - **35 Engineered Features**: Generates all required features from just 6 base columns
  - Automatic preprocessing with scaler
  - Model performance metrics (R2: 0.747, RMSE: 0.541, MAE: 0.227)
  - Download predictions as CSV

### Required Columns for Predictions

**⚠️ Time Series Model:** This model uses **sequences of 10 consecutive timesteps** to make predictions.
- Need **at least 10 rows** of data to generate 1 prediction
- From N rows, you'll get **N-9 predictions** (due to sequence windowing)
- Each prediction uses information from the previous 10 time periods

Your dataset needs these **6 base columns** (the app will auto-engineer the rest):
- `nca` - Natural Capital Accounting value
- `gdp` - Gross Domestic Product
- `growth_gdp` - GDP growth rate
- `growth_nca` - NCA growth rate  
- `loggrowth_gdp` - Log of GDP growth
- `loggrowth_nca` - Log of NCA growth

**Optional columns** (recommended for time-series data):
- `year` - For temporal sorting
- `code` or `countryname` - For grouping by entity

The application will automatically:
1. Engineer 35 features (lag features, rolling means, intensity ratios)
2. Scale features using pre-fitted scaler
3. Create sequences of 10 timesteps
4. Generate predictions using XGBoost model

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Streamlit Interface (Recommended)

Run the Streamlit web interface:

```bash
streamlit run app_streamlit.py
```

The application will be available at `http://localhost:8501`

This provides an easy-to-use web interface with:
- Drag-and-drop file upload
- Automatic EDA generation with visualizations
- One-click predictions with downloadable results
- Beautiful, responsive UI
- Python 3.13 compatible

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

## API Endpoints

### 1. Upload & Generate EDA
**POST** `/upload-eda`

Upload a CSV or Excel file to automatically generate exploratory data analysis.

**Request:**
- File: CSV or Excel (.csv, .xlsx, .xls)

**Response:**
- Session ID
- Dataset information (rows, columns, dtypes)
- Summary statistics
- Missing values analysis
- List of numerical and categorical columns
- Paths to generated plots

**Example (using curl):**
```bash
curl -X POST "http://127.0.0.1:8005/upload-eda" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_dataset.csv"
```

### 2. Make Predictions
**POST** `/predict`

Upload a dataset to generate predictions using the pre-trained XGBoost model.

**Request:**
- File: CSV or Excel (.csv, .xlsx, .xls)

**Response:**
- Session ID
- Number of predictions
- Array of predictions
- Model information and metrics
- Download URL for results

**Example (using curl):**
```bash
curl -X POST "http://127.0.0.1:8005/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_dataset.csv"
```

### 3. Get Model Information
**GET** `/model-info`

Retrieve information about the loaded XGBoost model.

**Response:**
- Model name
- Performance metrics (R2, RMSE, MAE, MSE)
- Number of features
- Deployment date

### 4. Get Plot Image
**GET** `/plots/{filename}`

Retrieve generated plot images.

### 5. Download Prediction Results
**GET** `/download/{filename}`

Download prediction results as CSV file.

### 6. Health Check
**GET** `/health`

Check API health and model loading status.

## Usage Example

### Using Streamlit Interface (Easiest)

1. Run `streamlit run app_streamlit.py`
2. Browser will automatically open to `http://localhost:8501`
3. Upload your CSV/Excel file
4. View automatic EDA generation or make predictions
5. Download prediction results

### Using Python with FastAPI Backend

```python
import requests

# Upload dataset and get EDA
with open('dataset.csv', 'rb') as f:
    response = requests.post('http://127.0.0.1:8005/upload-eda', 
                           files={'file': f})
    eda_result = response.json()
    print(f"Session ID: {eda_result['session_id']}")
    print(f"Dataset shape: {eda_result['dataset_info']['n_rows']} rows, "
          f"{eda_result['dataset_info']['n_columns']} columns")
    
    # Download plots
    for plot_name, plot_path in eda_result['plots'].items():
        plot_filename = plot_path.split('/')[-1]
        plot_response = requests.get(f'http://127.0.0.1:8005/plots/{plot_filename}')
        with open(f'downloaded_{plot_name}.png', 'wb') as img:
            img.write(plot_response.content)

# Make predictions
with open('dataset.csv', 'rb') as f:
    response = requests.post('http://127.0.0.1:8005/predict',
                           files={'file': f})
    prediction_result = response.json()
    print(f"Predictions: {prediction_result['predictions'][:5]}...")  # First 5
    
    # Download results
    download_url = prediction_result['download_url']
    results = requests.get(f'http://127.0.0.1:8005{download_url}')
    with open('predictions.csv', 'wb') as f:
        f.write(results.content)
```

## Project Structure

```plaintext
eda-fastapi/
├── app_streamlit.py          # Streamlit web interface (main app)
├── main.py                   # FastAPI application
├── eda_module.py             # EDA generation logic
├── prediction.py             # Model prediction logic
├── models.py                 # Pydantic schemas
├── requirements.txt          # Dependencies
├── uploads/                  # Uploaded files (temporary)
├── plots/                    # Generated visualizations
├── results/                  # Prediction results
├── static/                   # HTML/CSS/JS frontend (optional)
└── models_deployment/        # Pre-trained models
    ├── xgboost_model.pkl
    ├── scaler.pkl
    └── model_performance.json
```

## Model Information

- **Model Type**: XGBoost Regressor
- **Features**: 35 numerical features
- **Performance Metrics**:
  - R² Score: 0.7475
  - RMSE: 0.541
  - MAE: 0.227
  - MSE: 0.293

## Notes

- Supported file formats: CSV, Excel (.xlsx, .xls)
- Plots are automatically generated for numerical and categorical features
- Uploaded files are stored temporarily and can be cleaned periodically
- The model expects numerical features and will automatically scale them using the pre-trained scaler
