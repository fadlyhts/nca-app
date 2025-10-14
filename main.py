from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import uuid
import shutil
from typing import Optional

from models import EDAResponse, PredictionResponse, ModelInfoResponse
from eda_module import EDAGenerator
from prediction_lstm import LSTMPredictor

app = FastAPI(
    title="EDA & Prediction API",
    description="Automated Exploratory Data Analysis and LSTM Time Series Prediction Pipeline",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
PLOTS_DIR = Path("plots")
RESULTS_DIR = Path("results")
STATIC_DIR = Path("static")

UPLOAD_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

predictor = LSTMPredictor()


def load_dataframe(file_path: Path) -> pd.DataFrame:
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return {
        "message": "Welcome to EDA & Prediction API",
        "endpoints": {
            "upload_eda": "/upload-eda",
            "predict": "/predict",
            "model_info": "/model-info",
            "plots": "/plots/{filename}",
            "download": "/download/{filename}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor.model is not None}


@app.post("/upload-eda", response_model=EDAResponse)
async def upload_and_generate_eda(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload CSV or Excel file."
        )
    
    session_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{session_id}{file_extension}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        df = load_dataframe(file_path)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        eda_generator = EDAGenerator(df, session_id, str(PLOTS_DIR))
        eda_result = eda_generator.get_complete_eda()
        
        return EDAResponse(**eda_result)
    
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Make LSTM time series predictions on uploaded dataset.
    
    Predicts intensity_nca for the next 5 years using the last 5 years of data.
    """
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload CSV or Excel file."
        )
    
    session_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{session_id}_predict{file_extension}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        df = load_dataframe(file_path)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        predictions, start_years, model_info = predictor.predict(df)
        
        predictions_df = predictor.format_predictions_as_dataframe(
            predictions, start_years, df
        )
        
        result_path = RESULTS_DIR / f"{session_id}_predictions.csv"
        predictor.save_predictions(predictions_df, str(result_path))
        
        return {
            "session_id": session_id,
            "n_sequences": len(predictions),
            "predictions_shape": predictions.shape,
            "lookback": model_info['lookback'],
            "horizon": model_info['horizon'],
            "target": model_info['target'],
            "predictions_summary": {
                "min": float(predictions.min()),
                "max": float(predictions.max()),
                "mean": float(predictions.mean())
            },
            "model_info": model_info,
            "download_url": f"/download/{session_id}_predictions.csv",
            "message": f"Generated {len(predictions)} sequences, each predicting {model_info['horizon']} future years"
        }
    
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    try:
        info = predictor.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")


@app.get("/model-features")
async def get_model_features():
    try:
        info = predictor.get_model_info()
        return {
            "required_features": info['features'],
            "n_features": info['n_features'],
            "lookback": info['lookback'],
            "horizon": info['horizon'],
            "target": info['target'],
            "description": info['description']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving features: {str(e)}")


@app.get("/plots/{filename}")
async def get_plot(filename: str):
    file_path = PLOTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(file_path, media_type="image/png")


@app.get("/download/{filename}")
async def download_results(filename: str):
    file_path = RESULTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
