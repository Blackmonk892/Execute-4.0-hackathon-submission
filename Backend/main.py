from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Optional


class TransactionData(BaseModel):
    transaction_id: float
    customer_id: float
    merchant_id: float
    amount: float
    transaction_time: str
    customer_age: float
    card_type: str  
    location: str  
    purchase_category: str  
    fraud_type: str 

app = FastAPI()


model = joblib.load(r"C:\Users\USER\Desktop\HACKATHONS\e-cell hackathon\Models\rf_fraud_model.pkl")

expected_features = model.feature_names_in_

@app.post("/predict/")
def predict_fraud(transaction_data: TransactionData):
    try:
        
        df = pd.DataFrame([transaction_data.dict()])
        
        
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        df['day'] = df['transaction_time'].dt.day
        df['hour'] = df['transaction_time'].dt.hour
        df['month'] = df['transaction_time'].dt.month
        df['weekday'] = df['transaction_time'].dt.dayofweek
        
        
        df = df.drop('transaction_time', axis=1)
        
        
        
        card_type_columns = ['card_type_MasterCard', 'card_type_Rupay', 'card_type_Visa']
        for col in card_type_columns:
            card = col.split('_')[1]
            df[col] = (df['card_type'] == card).astype(int)
        
        
        location_columns = ['location_Ahmedabad', 'location_Bangalore', 'location_Chennai', 
                           'location_Delhi', 'location_Hyderabad', 'location_Jaipur', 
                           'location_Kolkata', 'location_Mumbai', 'location_Pune', 'location_Surat']
        for col in location_columns:
            loc = col.split('_')[1]
            df[col] = (df['location'] == loc).astype(int)
        
        
        purchase_columns = ['purchase_category_Digital', 'purchase_category_POS']
        for col in purchase_columns:
            cat = col.split('_')[1]
            df[col] = (df['purchase_category'] == cat).astype(int)
        
        
        fraud_type_columns = ['fraud_type_Identity theft', 'fraud_type_Malware', 
                             'fraud_type_Payment card fraud', 'fraud_type_phishing', 
                             'fraud_type_scam']
        for col in fraud_type_columns:
            fraud = '_'.join(col.split('_')[1:])
            df[col] = (df['fraud_type'] == fraud).astype(int)
        
        
        df = df.drop(['card_type', 'location', 'purchase_category', 'fraud_type'], axis=1)
        
        
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        
        
        df = df[expected_features]
        
        
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1] 

        return {
            "prediction": int(prediction[0]),
            "is_fraud": bool(prediction[0]),
            "fraud_probability": float(probability),
            "transaction_id": transaction_data.transaction_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/")
def read_root():
    return {
        "message": "Fraud Detection API",
        "endpoints": {
            "/predict": "POST - Predict if a transaction is fraudulent",
            "/health": "GET - Check API health"
        },
        "model": "Random Forest Classifier"
    }
