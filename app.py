import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛒 Sales Prediction App")

# Sidebar for file upload
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded!")
    
    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("🎯 Select Features and Target")
    feature_cols = st.multiselect("Select Feature Columns (numeric only):", numeric_cols)
    target_col = st.selectbox("Select Target Column:", numeric_cols)

    if feature_cols and target_col:
        X = df[feature_cols].dropna()
        y = df[target_col].loc[X.index]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        st.sidebar.header("📈 Model Metrics")
        st.sidebar.metric("R² Score", f"{r2:.3f}")
        st.sidebar.metric("MAE", f"{mae:.2f}")
        st.sidebar.metric("RMSE", f"{rmse:.2f}")

        st.subheader("📉 Prediction Results")
        results_df = pd.DataFrame({"Actual": y, "Predicted": y_pred})
        st.dataframe(results_df.head())

        # Plotting
        st.subheader("📍 Prediction Plot")
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, edgecolor='k', facecolor='skyblue', label="Predicted Points")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal Line (y = x)")
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Please select at least one feature and a target column.")

else:
    st.info("👈 Upload a CSV file to get started.")
