import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# App Title
st.title("📦 Amazon AI Hackathon: Supply Chain & Inventory Agent")
st.write("An intelligent demand forecasting and inventory optimization agent for third-party sellers.")

# 1. Generate Model Data & Train (Cached for speed)
@st.cache_resource
def train_model():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="W")
    products = ["Winter_Jacket", "Running_Shoes", "Smart_Watch"]
    regions = ["Seattle_FC", "Austin_FC", "NewYork_FC"]
    
    data = []
    for product in products:
        for region in regions:
            for date in dates:
                base = 100 if product == "Winter_Jacket" else 200
                seasonality = 30 * np.sin(2 * np.pi * date.dayofyear / 365)
                weather_factor = 20 if "Seattle" in region and product == "Winter_Jacket" and date.month in [10, 11, 12, 1, 2] else 0
                noise = np.random.normal(0, 10)
                units_sold = max(10, int(base + seasonality + weather_factor + noise))
                data.append({
                    "date": date, "product": product, "region": region,
                    "month": date.month, "week_of_year": date.isocalendar().week, "units_sold": units_sold
                })
                
    df = pd.DataFrame(data)
    df['lag_1_week'] = df.groupby(['product', 'region'])['units_sold'].shift(1).bfill()
    df['rolling_mean_4w'] = df.groupby(['product', 'region'])['units_sold'].transform(lambda x: x.rolling(4, min_periods=1).mean())
    
    df_model = pd.get_dummies(df, columns=['product', 'region'], drop_first=True)
    target = 'units_sold'
    features = [f for f in df_model.columns if f not in ['date', target]]
    
    X = df_model[features]
    y = df_model[target]
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return df, model, features

df_sales, model, features = train_model()

# 2. User Inputs via Sidebar / Form
st.sidebar.header("Seller Parameters")
selected_product = st.sidebar.selectbox("Select Product", ["Winter_Jacket", "Running_Shoes", "Smart_Watch"])
selected_region = st.sidebar.selectbox("Select Fulfillment Center (FC)", ["Seattle_FC", "Austin_FC", "NewYork_FC"])
current_stock = st.sidebar.number_input("Current Warehouse Stock", min_value=0, value=50)
planned_shipment = st.sidebar.number_input("Planned Shipment Units", min_value=0, value=300)

if st.button("Run AI Agent Recommendation"):
    subset = df_sales[(df_sales['product'] == selected_product) & (df_sales['region'] == selected_region)]
    latest_row = subset.iloc[-1]
    
    input_data = {
        'month': latest_row['month'],
        'week_of_year': (latest_row['week_of_year'] % 52) + 1,
        'lag_1_week': latest_row['units_sold'],
        'rolling_mean_4w': subset['units_sold'].tail(4).mean()
    }
    
    input_df = pd.DataFrame([input_data])
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    prod_col = f"product_{selected_product}"
    reg_col = f"region_{selected_region}"
    if prod_col in input_df.columns: input_df[prod_col] = 1
    if reg_col in input_df.columns: input_df[reg_col] = 1
    
    predicted_demand = int(model.predict(input_df[features])[0])
    projected_inventory = current_stock + planned_shipment
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Demand", f"{predicted_demand} units")
    col2.metric("Projected Inventory", f"{projected_inventory} units")
    col3.metric("Stock Difference", f"{projected_inventory - predicted_demand} units")
    
    # Agent Decision Logic
    st.subheader("🤖 Agent Recommendation")
    if projected_inventory > predicted_demand * 1.3:
        excess = projected_inventory - predicted_demand
        st.warning(f"⚠️ *OVERSTOCK RISK:* You are sending {excess} too many units. Consider reducing the shipment to avoid long-term storage fees at the Amazon fulfillment center.")
    elif projected_inventory < predicted_demand:
        shortage = predicted_demand - projected_inventory
        st.error(f"🚨 *STOCK-OUT RISK:* You are short by ~{shortage} units based on local demand trends and weather spikes. Increase your shipment size!")
    else:
        st.success("✅ *OPTIMAL:* Your planned shipment perfectly balances projected demand.")
