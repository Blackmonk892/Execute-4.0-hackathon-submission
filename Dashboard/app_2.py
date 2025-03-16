import streamlit as st

# This must be the first Streamlit command
st.set_page_config(
    page_title="Advanced Fraud Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random
import pydeck as pdk
import altair as alt

base_url = "http://127.0.0.1:8000"

# Try importing streamlit_option_menu, with error handling
try:
    from streamlit_option_menu import option_menu
except ImportError:
    st.error("Missing dependency: streamlit-option-menu. Please install it with `pip install streamlit-option-menu`")
    option_menu = None

# Custom CSS with modern design elements
st.markdown("""
<style>
    /* Main styling with modern aesthetics */
    .main-header {font-size: 2.8rem; background: linear-gradient(90deg, #1E3A8A, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1.5rem; font-weight: 800; letter-spacing: -0.5px;}
    .sub-header {font-size: 1.6rem; color: #1E3A8A; margin-top: 1rem; margin-bottom: 0.5rem; font-weight: 600;}
    
    /* Modern card designs with subtle shadows and gradients */
    .card {border-radius: 12px; padding: 22px; background-color: #ffffff; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05); margin-bottom: 24px; border-left: none; transition: transform 0.3s ease, box-shadow 0.3s ease;}
    .card:hover {transform: translateY(-5px); box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);}
    
    /* Alert styles with animations */
    .fraud-alert {background: linear-gradient(135deg, #FFEBEE, #FFCDD2); color: #C62828; padding: 18px; border-radius: 10px; font-weight: bold; text-align: center; font-size: 1.2rem; animation: pulse 2s infinite; box-shadow: 0 5px 15px rgba(198, 40, 40, 0.2);}
    .safe-alert {background: linear-gradient(135deg, #E8F5E9, #C8E6C9); color: #2E7D32; padding: 18px; border-radius: 10px; font-weight: bold; text-align: center; font-size: 1.2rem; box-shadow: 0 5px 15px rgba(46, 125, 50, 0.2);}
    
    /* Modern metric cards */
    .metric-card {text-align: center; padding: 20px; border-radius: 12px; background: linear-gradient(145deg, #f0f2f6, #ffffff); box-shadow: 0 5px 15px rgba(0,0,0,0.05); transition: transform 0.3s ease;}
    .metric-card:hover {transform: translateY(-5px);}
    
    /* Insight cards with gradient borders */
    .insight-card {background-color: #ffffff; border-left: none; border-radius: 10px; padding: 18px; margin-bottom: 18px; position: relative; box-shadow: 0 5px 15px rgba(59, 130, 246, 0.1); overflow: hidden;}
    .insight-card::before {content: ""; position: absolute; left: 0; top: 0; height: 100%; width: 5px; background: linear-gradient(to bottom, #3B82F6, #1E3A8A);}
    
    /* Risk indicators */
    .risk-high {color: #C62828; font-weight: bold; background: rgba(198, 40, 40, 0.1); padding: 5px 10px; border-radius: 20px;}
    .risk-medium {color: #FF8F00; font-weight: bold; background: rgba(255, 143, 0, 0.1); padding: 5px 10px; border-radius: 20px;}
    .risk-low {color: #2E7D32; font-weight: bold; background: rgba(46, 125, 50, 0.1); padding: 5px 10px; border-radius: 20px;}
    
    /* Animations */
    @keyframes pulse {
        0% {box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.4);}
        70% {box-shadow: 0 0 0 10px rgba(198, 40, 40, 0);}
        100% {box-shadow: 0 0 0 0 rgba(198, 40, 40, 0);}
    }
    
    /* Modern tab styling */
    .stTabs [data-baseweb="tab-list"] {gap: 2px; background-color: #f8f9fa; border-radius: 12px; padding: 5px;}
    .stTabs [data-baseweb="tab"] {border-radius: 10px; padding: 12px 18px; font-weight: 600; transition: all 0.3s ease;}
    .stTabs [aria-selected="true"] {background: linear-gradient(90deg, #1E3A8A, #3B82F6); color: white; box-shadow: 0 4px 10px rgba(30, 58, 138, 0.3);}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {width: 8px; height: 8px;}
    ::-webkit-scrollbar-track {background: #f1f1f1; border-radius: 10px;}
    ::-webkit-scrollbar-thumb {background: #1E3A8A; border-radius: 10px;}
    ::-webkit-scrollbar-thumb:hover {background: #3B82F6;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {font-size: 2rem;}
        .sub-header {font-size: 1.4rem;}
    }
    
    /* Tooltip styling */
    .tooltip {position: relative; display: inline-block; cursor: help;}
    .tooltip .tooltiptext {visibility: hidden; width: 200px; background-color: #1E3A8A; color: white; text-align: center; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s;}
    .tooltip:hover .tooltiptext {visibility: visible; opacity: 1;}
    
    /* Custom button styling */
    .stButton>button {background: linear-gradient(90deg, #1E3A8A, #3B82F6); color: white; border: none; border-radius: 8px; padding: 10px 24px; font-weight: 600; transition: all 0.3s ease;}
    .stButton>button:hover {transform: translateY(-2px); box-shadow: 0 5px 15px rgba(30, 58, 138, 0.3);}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'fraud_trends' not in st.session_state:
    # Generate sample fraud trend data
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    st.session_state.fraud_trends = pd.DataFrame({
        'date': dates,
        'fraud_rate': [random.uniform(1, 5) for _ in range(30)],
        'avg_fraud_amount': [random.uniform(500, 2000) for _ in range(30)]
    })
if 'fraud_types' not in st.session_state:
    st.session_state.fraud_types = {
        'Identity theft': random.randint(20, 100),
        'Malware': random.randint(10, 50),
        'Payment card fraud': random.randint(30, 120),
        'Phishing': random.randint(40, 90),
        'Scam': random.randint(15, 70)
    }
if 'recent_alerts' not in st.session_state:
    st.session_state.recent_alerts = [
        {"time": "10:15 AM", "type": "High Risk", "message": "Unusual transaction pattern detected", "amount": "$2,450.00"},
        {"time": "09:30 AM", "type": "Medium Risk", "message": "Multiple transactions from new location", "amount": "$1,200.00"},
        {"time": "Yesterday", "type": "Low Risk", "message": "Transaction amount above average", "amount": "$850.00"},
    ]

# API connection check with caching
@st.cache_data(ttl=300)
def check_api_connection(base_url):
    try:
        return True  # Simulated connection for demo
    except:
        return False

# Generate location coordinates for map visualization
@st.cache_data(ttl=3600)
def get_location_coordinates(location):
    # Dummy coordinates for demonstration
    coordinates = {
        "Mumbai": [19.0760, 72.8777],
        "Delhi": [28.6139, 77.2090],
        "Bangalore": [12.9716, 77.5946],
        "Chennai": [13.0827, 80.2707],
        "Hyderabad": [17.3850, 78.4867],
        "Kolkata": [22.5726, 88.3639],
        "Pune": [18.5204, 73.8567],
        "Ahmedabad": [23.0225, 72.5714],
        "Jaipur": [26.9124, 75.7873],
        "Surat": [21.1702, 72.8311]
    }
    return coordinates.get(location, [20.5937, 78.9629])

# Calculate risk score based on multiple factors
def calculate_risk_score(amount, hour, age, category, card_type, location):
    score = 0
    
    # Amount factor
    if amount > 5000:
        score += 30
    elif amount > 2000:
        score += 15
    
    # Time factor
    if hour < 6 or hour > 22:
        score += 25
    
    # Age factor
    if age < 25:
        score += 15
    
    # Category factor
    if category == "Digital":
        score += 10
    
    # Card type factor
    if card_type == "Visa":
        score += 5
    
    # Location factor - some locations might be higher risk
    high_risk_locations = ["Mumbai", "Delhi"]
    if location in high_risk_locations:
        score += 15
    
    return min(score, 100)  # Cap at 100

# Function to get risk category
def get_risk_category(score):
    if score >= 70:
        return "High", "risk-high"
    elif score >= 40:
        return "Medium", "risk-medium"
    else:
        return "Low", "risk-low"

# Generate AI insights based on transaction
@st.cache_data(ttl=3600)
def generate_ai_insights(transaction_data, fraud_probability):
    insights = []
    
    # Amount insights
    if transaction_data["amount"] > 5000:
        insights.append("Transaction amount is significantly higher than average for this customer profile.")
    
    # Time insights
    hour = int(transaction_data["transaction_time"].split(" ")[1].split(":")[0])
    if hour < 6 or hour > 22:
        insights.append("Transaction occurred during unusual hours (late night/early morning).")
    
    # Location insights
    if transaction_data["location"] != "Bangalore":
        insights.append(f"Transaction location ({transaction_data['location']}) differs from customer's usual activity area.")
    
    # Category insights
    if transaction_data["purchase_category"] == "Digital" and fraud_probability > 0.3:
        insights.append("Digital purchases have higher fraud rates in this transaction amount range.")
    
    # Pattern insights
    if fraud_probability > 0.5:
        insights.append("Transaction matches patterns seen in recent fraud cases.")
    
    # Add a general insight if we have few specific ones
    if len(insights) < 2:
        insights.append("Transaction parameters are within normal ranges for this customer profile.")
    
    return insights

# Create fraud gauge chart
def create_fraud_gauge(fraud_percentage):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fraud_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Fraud Risk", 'font': {'size': 24, 'color': '#1E3A8A'}},
        delta={'reference': fraud_percentage-5, 'increasing': {'color': "#C62828"}, 'decreasing': {'color': "#2E7D32"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E3A8A"},
            'bar': {'color': "#1E3A8A"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#1E3A8A",
            'steps': [
                {'range': [0, 30], 'color': "#E8F5E9"},
                {'range': [30, 70], 'color': "#FFF8E1"},
                {'range': [70, 100], 'color': "#FFEBEE"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': fraud_percentage
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1E3A8A", 'family': "Arial"}
    )
    return fig

# Create trend visualization
def create_trend_visualization(trend_data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=trend_data['date'], 
            y=trend_data['fraud_rate'], 
            name="Fraud Rate (%)",
            line=dict(color='#1E3A8A', width=3),
            fill='tozeroy',
            fillcolor='rgba(30, 58, 138, 0.1)'
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=trend_data['date'], 
            y=trend_data['avg_fraud_amount'], 
            name="Avg Fraud Amount ($)",
            line=dict(color='#C62828', width=3, dash='dot'),
        ),
        secondary_y=True,
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Date", showgrid=False)
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=False, showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="Average Fraud Amount ($)", secondary_y=True, showgrid=False)
    
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1E3A8A", 'family': "Arial"}
    )
    
    return fig

# Create fraud type distribution chart
def create_fraud_type_chart():
    labels = list(st.session_state.fraud_types.keys())
    values = list(st.session_state.fraud_types.values())
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels, 
            values=values,
            hole=.4,
            textinfo='label+percent',
            marker=dict(
                colors=['#1E3A8A', '#3B82F6', '#93C5FD', '#BFDBFE', '#DBEAFE'],
                line=dict(color='#FFFFFF', width=2)
            )
        )
    ])
    
    fig.update_layout(
        title="Fraud Type Distribution",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1E3A8A", 'family': "Arial"},
        showlegend=False
    )
    
    return fig

# Create map visualization
def create_map_visualization(transactions):
    if not transactions:
        return None
    
    # Create dataframe with location data
    df = pd.DataFrame(transactions)
    df['coordinates'] = df['location'].apply(get_location_coordinates)
    df['latitude'] = df['coordinates'].apply(lambda x: x[0])
    df['longitude'] = df['coordinates'].apply(lambda x: x[1])
    df['color'] = df['is_fraud'].apply(lambda x: [200, 30, 0, 160] if x else [0, 100, 0, 160])
    df['size'] = df['amount'].apply(lambda x: min(x/100, 50))
    
    # Create map
    view_state = pdk.ViewState(
        latitude=20.5937,
        longitude=78.9629,
        zoom=4,
        pitch=50
    )
    
    # Define layers
    layers = [
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position=['longitude', 'latitude'],
            get_color='color',
            get_radius='size',
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=6,
            radius_min_pixels=5,
            radius_max_pixels=30,
            line_width_min_pixels=1
        )]