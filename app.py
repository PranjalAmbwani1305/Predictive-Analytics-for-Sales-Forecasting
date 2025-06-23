import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="Sales Prediction App", layout="wide")

# Sidebar - Upload Data
st.sidebar.title("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded!")
    
    st.title("🛒 Sales Prediction App")
    
    # Show preview of data
    st.subheader("🔍 Preview Data")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Select features and target
    st.subheader("🎯 Select Features and Target")
    feature_cols = st.multiselect("Select Feature Columns (numeric only):", options=numeric_columns)
    target_col = st.selectbox("Select Target Column:", options=numeric_columns)
    
    if feature_cols and target_col:
        # Split data
        X = df[feature_cols]
        y = df[target_col]
        
        # Model training
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Display metrics
        st.sidebar.subheader("📈 Model Metrics")
        st.sidebar.write("**R² Score**:", round(r2, 4))
        st.sidebar.write("**MAE**:", round(mae, 2))
        st.sidebar.write("**RMSE**:", round(rmse, 2))
        
        # Show Prediction Results
        st.subheader("📊 Prediction Results")
        result_df = df[[target_col]].copy()
        result_df["Predicted_" + target_col] = y_pred
        st.dataframe(result_df.head(15))
        
        # Plot: Actual vs Predicted
        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, edgecolors='black', alpha=0.7, label="Predicted Points")
        ax.plot(y, y, 'r--', label="Ideal Line (y = x)")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Sales")
        ax.legend()
        st.pyplot(fig)
