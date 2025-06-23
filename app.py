import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded!")

    # Encode categorical columns
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

    st.markdown("## 🛒 Sales Prediction App")
    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    st.markdown("## 🎯 Select Features and Target")

    numeric_cols = df_encoded.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select Target Column:", numeric_cols)

    feature_candidates = [col for col in numeric_cols if col != target]
    selected_features = st.multiselect("Select Feature Columns:", feature_candidates)

    if selected_features and target:
        X = df_encoded[selected_features]
        y = df_encoded[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        with st.sidebar:
            st.subheader("📈 Model Metrics")
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")

        st.markdown("## 📊 Actual vs Predicted Graph")
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.6, color='skyblue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        st.markdown("## 📋 Prediction Results")
        results_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions
        })
        st.dataframe(results_df.style.highlight_max(axis=0, subset=["Predicted"], color='lightgreen'))
    else:
        st.warning("⚠️ Please select both feature(s) and target column.")
else:
    st.info("📁 Upload a CSV file to get started.")
