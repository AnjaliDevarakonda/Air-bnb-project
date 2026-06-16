import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="London Airbnb Market Analysis", layout="wide")

st.title("🏠 London Airbnb Market Analysis")
st.markdown("Explore listings, analyze hosts, and predict prices")

# Load data
@st.cache_data
def load_data():
    # Try multiple paths
    possible_paths = [
        'listings.csv',
        'data/listings.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        st.error("Dataset not found. Please make sure 'listings.csv' is in the data folder.")
        return pd.DataFrame()
    
    df = df.dropna(subset=['price', 'latitude', 'longitude'])
    df = df[df['price'] > 0]
    df = df[df['price'] < 1000]
    return df

df = load_data()

if df.empty:
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    neighborhoods = ['All'] + sorted(df['neighbourhood'].unique().tolist())
    selected_neighborhood = st.selectbox("Select Neighbourhood", neighborhoods)
    
    room_types = ['All'] + sorted(df['room_type'].unique().tolist())
    selected_room_type = st.selectbox("Select Room Type", room_types)
    
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    price_range = st.slider("Price Range (£)", min_price, max_price, (min_price, max_price))

# Filter data
filtered_df = df.copy()
if selected_neighborhood != 'All':
    filtered_df = filtered_df[filtered_df['neighbourhood'] == selected_neighborhood]
if selected_room_type != 'All':
    filtered_df = filtered_df[filtered_df['room_type'] == selected_room_type]
filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Listings", f"{len(filtered_df):,}")
col2.metric("Average Price", f"£{filtered_df['price'].mean():.2f}")
col3.metric("Median Price", f"£{filtered_df['price'].median():.2f}")
col4.metric("Unique Neighbourhoods", len(filtered_df['neighbourhood'].unique()))

# Map
st.subheader("🗺️ Listings Map")
fig = px.scatter_mapbox(
    filtered_df,
    lat='latitude',
    lon='longitude',
    color='price',
    hover_name='name',
    hover_data={'price': True, 'room_type': True, 'neighbourhood': True},
    zoom=10,
    height=500,
    color_continuous_scale='Viridis',
    mapbox_style='open-street-map'
)
st.plotly_chart(fig, use_container_width=True)

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price by Neighbourhood")
    avg_price = filtered_df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(15)
    fig = px.bar(avg_price, title="Average Price by Neighbourhood")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏷️ Room Type Distribution")
    room_type_counts = filtered_df['room_type'].value_counts()
    fig = px.pie(room_type_counts, values='count', names=room_type_counts.index)
    st.plotly_chart(fig, use_container_width=True)

# Host analysis
st.divider()
st.subheader("👤 Host Analysis")

col1, col2 = st.columns(2)

with col1:
    top_hosts = filtered_df['host_name'].value_counts().head(10)
    fig = px.bar(top_hosts, title="Top 10 Busiest Hosts")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    host_listing_count = filtered_df.groupby('host_name').size().reset_index(name='listing_count')
    single_hosts = host_listing_count[host_listing_count['listing_count'] == 1]
    multi_hosts = host_listing_count[host_listing_count['listing_count'] > 1]
    
    fig = px.pie(
        values=[len(single_hosts), len(multi_hosts)],
        names=['Single Property Hosts', 'Multi-Property Hosts'],
        title="Hosts with Multiple Listings"
    )
    st.plotly_chart(fig, use_container_width=True)

# Price Prediction
st.divider()
st.header("🔮 Predict Listing Price")

# Try loading trained model
@st.cache_resource
def load_model():
    model_paths = [
        '../models/best_model.pkl',
        'models/best_model.pkl'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            scaler = joblib.load('../models/scaler.pkl')
            le_neighborhood = joblib.load('../models/le_neighborhood.pkl')
            le_room_type = joblib.load('../models/le_room_type.pkl')
            return model, scaler, le_neighborhood, le_room_type, True
    
    return None, None, None, None, False

model, scaler, le_neighborhood, le_room_type, model_loaded = load_model()

col1, col2, col3 = st.columns(3)

with col1:
    pred_neighborhood = st.selectbox("Neighbourhood", sorted(df['neighbourhood'].unique()))
with col2:
    pred_room_type = st.selectbox("Room Type", sorted(df['room_type'].unique()))
with col3:
    pred_lat = st.number_input("Latitude", value=51.5, format="%.6f")
    pred_lon = st.number_input("Longitude", value=-0.12, format="%.6f")

if st.button("Predict Price", type="primary"):
    if model_loaded:
        try:
            # Prepare input
            neighborhood_code = le_neighborhood.transform([pred_neighborhood])[0]
            room_type_code = le_room_type.transform([pred_room_type])[0]
            
            # Calculate distance
            distance = np.sqrt((pred_lat - 51.5074)**2 + (pred_lon + 0.1278)**2)
            
            input_data = pd.DataFrame([{
                'latitude': pred_lat,
                'longitude': pred_lon,
                'neighborhood_code': neighborhood_code,
                'room_type_code': room_type_code,
                'distance_to_center': distance,
                'host_listing_count': 1
            }])
            
            # Scale
            numeric_cols = ['latitude', 'longitude', 'distance_to_center', 'host_listing_count']
            input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
            
            prediction = model.predict(input_data)[0]
            
            st.success(f"💰 Predicted Price: **£{prediction:.2f}**")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Train the model first by running the 02_Model_Training notebook.")

# Insights
st.divider()
st.subheader("💡 Key Business Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🏘️ Top Neighborhood Insights:**
    - Premium areas command higher prices
    - Central London neighborhoods have more listings
    - Room type varies significantly by area
    """)

with col2:
    st.markdown("""
    **📊 Market Trends:**
    - Multi-property hosts dominate premium areas
    - Private rooms are most common
    - Price range varies widely by location
    """)

# Data preview
with st.expander("📋 View Raw Data"):
    st.dataframe(filtered_df.head(100))
    st.caption(f"Showing {len(filtered_df)} out of {len(df)} total listings")