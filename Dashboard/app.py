import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; color: #1E3A8A; margin-top: 1rem; margin-bottom: 0.5rem;}
    .card {border-radius: 5px; padding: 20px; background-color: #f8f9fa; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;}
    .fraud-alert {background-color: #FFEBEE; color: #C62828; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;}
    .safe-alert {background-color: #E8F5E9; color: #2E7D32; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;}
    .metric-card {text-align: center; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🛡️ Fraud Detection System</h1>", unsafe_allow_html=True)

# Initialize session state for transaction history
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []

# API connection check
@st.cache_data(ttl=5)
def check_api_connection():
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Transaction Details</h2>", unsafe_allow_html=True)
    
    # API status indicator
    api_status = "🟢 Connected" if check_api_connection() else "🔴 Disconnected"
    st.markdown(f"**API Status**: {api_status}")
    
    # API endpoint configuration
    api_base_url = st.text_input("API URL", value="http://127.0.0.1:8000")
    
    # Transaction form
    with st.form("transaction_form"):
        transaction_id = st.number_input("Transaction ID", min_value=1, value=12345)
        customer_id = st.number_input("Customer ID", min_value=1, value=98765)
        merchant_id = st.number_input("Merchant ID", min_value=1, value=5432)
        amount = st.number_input("Amount ($)", min_value=0.01, value=1500.75, format="%.2f")
        
        # Date and time inputs
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date", datetime.now())
        with col2:
            time_val = st.time_input("Time", datetime.now().time())
        
        transaction_time = f"{date} {time_val.strftime('%H:%M')}"
        
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
        
        card_type = st.selectbox("Card Type", ["Visa", "MasterCard", "Rupay"])
        location = st.selectbox("Location", ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", 
                                            "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat"])
        purchase_category = st.selectbox("Purchase Category", ["Digital", "POS"])
        fraud_type = st.selectbox("Suspected Fraud Type", 
                                 ["Identity theft", "Malware", "Payment card fraud", "phishing", "scam"])
        
        submitted = st.form_submit_button("Check Fraud Risk")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Transaction Analysis</h2>", unsafe_allow_html=True)
    
    # Process form submission
    if submitted:
        if amount <= 0:
            st.error("Amount must be greater than zero")
        else:
            # Prepare data for API
            data = {
                "transaction_id": float(transaction_id),
                "customer_id": float(customer_id),
                "merchant_id": float(merchant_id),
                "amount": float(amount),
                "transaction_time": transaction_time,
                "customer_age": float(customer_age),
                "card_type": card_type,
                "location": location,
                "purchase_category": purchase_category,
                "fraud_type": fraud_type
            }
            
            # Call API
            try:
                response = requests.post(f"{api_base_url}/predict/", json=data, timeout=10)
                
                if response.status_code != 200:
                    st.error(f"API Error: Status code {response.status_code} - {response.text}")
                else:
                    result = response.json()
                    
                    # Validate response
                    if "is_fraud" not in result or "fraud_probability" not in result:
                        st.error(f"API response missing required fields. Response: {result}")
                    else:
                        # Add to transaction history
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        history_item = {
                            "timestamp": timestamp,
                            "transaction_id": transaction_id,
                            "amount": amount,
                            "is_fraud": result["is_fraud"],
                            "fraud_probability": result["fraud_probability"]
                        }
                        st.session_state.transaction_history.append(history_item)
                        
                        # Display result
                        if result["is_fraud"]:
                            st.markdown("<div class='fraud-alert'>⚠️ FRAUD DETECTED</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='safe-alert'>✅ TRANSACTION SAFE</div>", unsafe_allow_html=True)
                        
                        # Create gauge chart for fraud probability
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result["fraud_probability"] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Fraud Probability"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': result["fraud_probability"] * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display transaction details
                        st.markdown("<h3 class='sub-header'>Transaction Details</h3>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Transaction ID", f"#{transaction_id}")
                            st.metric("Amount", f"${amount:,.2f}")
                        with col2:
                            st.metric("Location", location)
                            st.metric("Card Type", card_type)
                        with col3:
                            st.metric("Purchase Type", purchase_category)
                            st.metric("Customer Age", customer_age)
                        
                        # Risk factors
                        st.markdown("<h3 class='sub-header'>Risk Factors</h3>", unsafe_allow_html=True)
                        risk_factors = []
                        
                        if amount > 5000:
                            risk_factors.append("High transaction amount")
                        
                        transaction_hour = int(time_val.strftime("%H"))
                        if transaction_hour < 6 or transaction_hour > 22:
                            risk_factors.append("Unusual transaction time")
                            
                        if customer_age < 25:
                            risk_factors.append("Young customer age")
                            
                        if purchase_category == "Digital":
                            risk_factors.append("Digital purchase (higher risk category)")
                        
                        if risk_factors:
                            for factor in risk_factors:
                                st.warning(factor)
                        else:
                            st.success("No significant risk factors identified")
            
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Unable to connect to the API server. Make sure it's running.")
            except json.JSONDecodeError:
                st.error("Error: Invalid response from API")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
    else:
        st.info("Fill in the transaction details and click 'Check Fraud Risk' to analyze")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Transaction history
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Transaction History</h2>", unsafe_allow_html=True)
    
    if st.session_state.transaction_history:
        history_df = pd.DataFrame(st.session_state.transaction_history)
        
        # Display metrics
        total_transactions = len(history_df)
        fraud_count = history_df['is_fraud'].sum()
        fraud_percentage = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0
        
        metrics_container = st.container()
        left_metric, right_metric = metrics_container.columns(2)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Transactions", total_transactions)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Fraud Detected", f"{fraud_percentage:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Create a pie chart for fraud vs legitimate
        labels = ['Legitimate', 'Fraudulent']
        values = [(total_transactions - fraud_count), fraud_count]
        colors = ['#4CAF50', '#F44336']
        
        fig = px.pie(values=values, names=labels, color_discrete_sequence=colors)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display transaction history table
        st.markdown("<h3 class='sub-header'>Recent Transactions</h3>", unsafe_allow_html=True)
        
        # Format the dataframe for display
        display_df = history_df.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
        display_df = display_df.sort_values('timestamp', ascending=False)
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x*100:.1f}%")
        display_df['is_fraud'] = display_df['is_fraud'].apply(lambda x: "🔴 Yes" if x else "🟢 No")
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
        display_df = display_df.rename(columns={
            'timestamp': 'Time',
            'transaction_id': 'ID',
            'amount': 'Amount',
            'is_fraud': 'Fraud',
            'fraud_probability': 'Risk'
        })
        
        st.dataframe(display_df[['Time', 'ID', 'Amount', 'Fraud', 'Risk']], hide_index=True, use_container_width=True)
    else:
        st.info("No transactions analyzed yet")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Additional information section
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>About the Fraud Detection System</h2>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["System Overview", "Model Performance", "Usage Guide"])

with tab1:
    st.markdown("""
    This fraud detection system uses a Random Forest machine learning model trained on historical transaction data to identify potentially fraudulent transactions. The model analyzes various transaction attributes including:
    
    - Transaction amount
    - Time of transaction
    - Customer demographics
    - Location
    - Purchase category
    - Card type
    
    The system provides real-time risk assessment with probability scores to help identify suspicious activities.
    """)

with tab2:
    st.markdown("### Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "99.14%")
    with col2:
        st.metric("Precision", "98%")
    with col3:
        st.metric("Recall", "99%")
    with col4:
        st.metric("AUC-ROC", "99.99%")
    
    st.markdown("""
    The confusion matrix shows excellent performance with minimal misclassifications:
    - True Negatives: 640
    - False Positives: 6
    - False Negatives: 2
    - True Positives: 284
    """)

with tab3:
    st.markdown("""
    ### How to Use This Dashboard
    
    1. Enter transaction details in the sidebar form
    2. Click "Check Fraud Risk" to analyze the transaction
    3. View the fraud probability and risk assessment
    4. Monitor transaction history and statistics
    
    ### Interpreting Results
    
    - **Fraud Probability**: Shows the likelihood of fraud from 0-100%
    - **Risk Factors**: Highlights specific elements of the transaction that may indicate higher risk
    - **Transaction History**: Tracks all analyzed transactions for monitoring patterns
    
    For suspicious transactions, consider additional verification steps before processing.
    """)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; color: #666;">
    <p>Fraud Detection System | Developed for E-Cell Hackathon</p>
</div>
""", unsafe_allow_html=True)
