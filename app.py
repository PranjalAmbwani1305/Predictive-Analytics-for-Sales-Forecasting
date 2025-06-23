import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Set app title
st.set_page_config(layout="wide")
st.title("🛒 Sales Prediction App")

# Upload CSV file
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")

    # Show preview
    st.subheader("🔍 Preview Data")
    st.dataframe(df.head())

    # Select feature and target
    st.subheader("🎯 Select Features and Target")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    feature_cols = st.multiselect("Select Feature Columns (numeric only):", options=numeric_cols)
    target_col = st.selectbox("Select Target Column:", options=numeric_cols)

    if feature_cols and target_col:
        # Clean data
        df_clean = df[feature_cols + [target_col]].dropna()
        X = df_clean[feature_cols]
        y = df_clean[target_col].astype(float)

        # Model training
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Evaluation
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Show metrics
        st.sidebar.subheader("📈 Model Metrics")
        st.sidebar.metric("R² Score", f"{r2:.3f}")
        st.sidebar.metric("MAE", f"{mae:.2f}")
        st.sidebar.metric("RMSE", f"{rmse:.2f}")

        # Prediction Results
        st.subheader("📊 Prediction Results")
        df_results = df_clean.copy()
        df_results["Predicted_" + target_col] = y_pred
        st.dataframe(df_results[[target_col] + feature_cols + ["Predicted_" + target_col]])

        # Plot Actual vs Predicted
        st.subheader("📉 Actual vs Predicted")

        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, color='skyblue', edgecolors='black', label="Predicted Points")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal Line (y = x)")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Sales")
        ax.legend()
        st.pyplot(fig)

else:
    st.info("⬆️ Please upload a CSV file to begin.")
