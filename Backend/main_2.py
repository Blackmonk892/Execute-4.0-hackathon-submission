from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, validator, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import joblib
import os
import logging
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the data model for input validation with enhanced validation
class TransactionData(BaseModel):
    transaction_id: float = Field(..., description="Unique identifier for the transaction")
    customer_id: float = Field(..., description="Unique identifier for the customer")
    merchant_id: float = Field(..., description="Unique identifier for the merchant")
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    transaction_time: str = Field(..., description="Transaction timestamp in format 'YYYY-MM-DD HH:MM'")
    customer_age: float = Field(..., ge=18, le=120, description="Age of the customer")
    card_type: str = Field(..., description="Type of card used for transaction")
    location: str = Field(..., description="Location where transaction occurred")
    purchase_category: str = Field(..., description="Category of purchase")
    fraud_type: str = Field(..., description="Type of potential fraud")
    
    # Add validation for transaction_time format
    @validator('transaction_time')
    def validate_transaction_time(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except:
            raise ValueError("transaction_time must be in a valid datetime format")
    
    # Add validation for card_type
    @validator('card_type')
    def validate_card_type(cls, v):
        valid_types = ["Visa", "MasterCard", "Rupay"]
        if v not in valid_types:
            raise ValueError(f"card_type must be one of {valid_types}")
        return v
    
    # Add validation for purchase_category
    @validator('purchase_category')
    def validate_purchase_category(cls, v):
        valid_categories = ["Digital", "POS"]
        if v not in valid_categories:
            raise ValueError(f"purchase_category must be one of {valid_categories}")
        return v

# Define the response model for better API documentation
class PredictionResponse(BaseModel):
    prediction: int
    is_fraud: bool
    fraud_probability: float
    transaction_id: float
    processing_time_ms: float

# Create FastAPI app with metadata
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions using machine learning",
    version="1.0.0"
)

# Cache the model loading to improve performance
@lru_cache(maxsize=1)
def get_model():
    """Load the model once and cache it for future use"""
    model_path = os.getenv("MODEL_PATH", r"C:\Users\USER\Desktop\HACKATHONS\e-cell hackathon\Models\best_rf_fraud_model.pkl")
    logger.info(f"Loading model from {model_path}")
    start_time = time.time()
    model = joblib.load(model_path)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return model

# Helper function to create one-hot encoded columns
def create_one_hot_columns(df: pd.DataFrame, column_name: str, values: List[str], prefix: str = None) -> pd.DataFrame:
    """
    Create one-hot encoded columns for categorical variables
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to encode
        values: List of possible values for the column
        prefix: Prefix to use for the new columns (defaults to column_name)
    
    Returns:
        DataFrame with one-hot encoded columns added
    """
    prefix = prefix or column_name
    for value in values:
        col_name = f"{prefix}_{value}"
        df[col_name] = (df[column_name] == value).astype(int)
    return df

# Dependency to get model and features
def get_model_and_features():
    """Dependency to get model and its features"""
    model = get_model()
    return {
        "model": model,
        "features": model.feature_names_in_
    }

@app.post("/predict/", response_model=PredictionResponse, summary="Predict fraud for a transaction")
def predict_fraud(
    transaction_data: TransactionData,
    model_data: Dict = Depends(get_model_and_features)
):
    """
    Predict whether a transaction is fraudulent
    
    - Takes transaction details as input
    - Returns prediction (0/1), boolean fraud indicator, and fraud probability
    """
    start_time = time.time()
    logger.info(f"Processing transaction {transaction_data.transaction_id}")
    
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([transaction_data.dict()])
        
        # Process transaction_time to extract temporal features
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        df['day'] = df['transaction_time'].dt.day
        df['hour'] = df['transaction_time'].dt.hour
        df['month'] = df['transaction_time'].dt.month
        df['weekday'] = df['transaction_time'].dt.dayofweek
        
        # Drop the original time column as it's not needed for prediction
        df = df.drop('transaction_time', axis=1)
        
        # Create one-hot encoded columns for all categorical variables
        # Card type encoding
        df = create_one_hot_columns(
            df, 'card_type', 
            ['MasterCard', 'Rupay', 'Visa'], 
            prefix='card_type'
        )
        
        # Location encoding
        df = create_one_hot_columns(
            df, 'location', 
            ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 
             'Jaipur', 'Kolkata', 'Mumbai', 'Pune', 'Surat'],
            prefix='location'
        )
        
        # Purchase category encoding
        df = create_one_hot_columns(
            df, 'purchase_category', 
            ['Digital', 'POS'], 
            prefix='purchase_category'
        )
        
        # Fraud type encoding
        df = create_one_hot_columns(
            df, 'fraud_type', 
            ['Identity theft', 'Malware', 'Payment card fraud', 'phishing', 'scam'], 
            prefix='fraud_type'
        )
        
        # Drop original categorical columns
        df = df.drop(['card_type', 'location', 'purchase_category', 'fraud_type'], axis=1)
        
        # Ensure all columns from training are present with default values
        model_features = model_data["features"]
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        
        # Ensure columns are in the same order as during training
        df = df[model_features]
        
        # Make prediction using the model
        model = model_data["model"]
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]  # Probability of fraud (class 1)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Log prediction result
        logger.info(
            f"Transaction {transaction_data.transaction_id} prediction: {bool(prediction[0])} "
            f"with probability {probability:.4f} in {processing_time:.2f}ms"
        )
        
        # Return prediction result
        return {
            "prediction": int(prediction[0]),
            "is_fraud": bool(prediction[0]),
            "fraud_probability": float(probability),
            "transaction_id": transaction_data.transaction_id,
            "processing_time_ms": processing_time
        }
    
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error for transaction {transaction_data.transaction_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    
    except Exception as e:
        # Handle other errors
        logger.error(f"Error processing transaction {transaction_data.transaction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health", summary="Check API health")
def health_check():
    """
    Health check endpoint to verify the API is running
    
    Returns:
        Dict with status information
    """
    try:
        # Try to load the model to ensure it's accessible
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "api_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/", summary="API information")
def read_root():
    """
    Root endpoint providing API documentation
    
    Returns:
        Dict with API information
    """
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict if a transaction is fraudulent",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        },
        "model": "Random Forest Classifier",
        "documentation": "Visit /docs for interactive API documentation"
    }
