import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")

    # Show Preview
    st.markdown("## 🛒 Sales Prediction App")
    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    # Feature & Target selection
    st.markdown("## 🎯 Select Features and Target")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    selected_features = st.multiselect("Select Feature Columns (numeric only):", numeric_cols)
    target = st.selectbox("Select Target Column:", numeric_cols)

    if selected_features and target:
        X = df[selected_features]
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # Metrics
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)

        with st.sidebar:
            st.subheader("📈 Model Metrics")
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")

        # Prediction Table
        st.markdown("## 📋 Prediction Results")
        result_df = df.copy()
        result_df["🔮 Prediction"] = predictions
        st.dataframe(result_df.style.highlight_max(axis=0, subset=["🔮 Prediction"], color='lightgreen'))

    else:
        st.warning("Please select both features and target to proceed.")
else:
    st.info("Upload a CSV file to get started.")
