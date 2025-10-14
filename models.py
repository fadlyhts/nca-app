from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class DatasetInfo(BaseModel):
    n_rows: int
    n_columns: int
    columns: List[str]
    dtypes: Dict[str, str]
    memory_usage: str


class MissingValuesInfo(BaseModel):
    column: str
    missing_count: int
    missing_percentage: float


class SummaryStatistics(BaseModel):
    statistics: Dict[str, Any]


class EDAResponse(BaseModel):
    session_id: str
    dataset_info: DatasetInfo
    summary_statistics: Dict[str, Any]
    missing_values: List[MissingValuesInfo]
    numerical_columns: List[str]
    categorical_columns: List[str]
    plots: Dict[str, str]
    message: str


class PredictionResponse(BaseModel):
    session_id: str
    n_predictions: int
    predictions: List[float]
    model_info: Dict[str, Any]
    download_url: Optional[str] = None
    message: str


class ModelInfoResponse(BaseModel):
    model_name: str
    metrics: Dict[str, float]
    n_features: int
    deployment_date: str
