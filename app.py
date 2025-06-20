import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set page config
st.set_page_config(page_title="📈 Sales Prediction Dashboard", layout="wide")

# Custom dark theme styling
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .stApp { background-color: #0e1117; color: white; }
        .css-1d391kg { background-color: #262730; }
        .stSelectbox>div>div>div>div, .stMultiSelect>div>div>div>div {
            background-color: #262730 !important;
            color: white !important;
        }
        .stMultiSelect>div>div>div>div>div>div {
            background-color: #ff4b4b !important;
            color: white !important;
        }
        .stSelectbox>div>div>div>div>div>div {
            color: white !important;
        }
        footer, #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
st.sidebar.title("🛠️ Upload & Configure")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])

# Load and process data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_columns = df.columns.tolist()

    feature_cols = st.sidebar.multiselect("Select feature columns:", options=all_columns)
    target_col = st.sidebar.selectbox("Select target column:", options=numeric_cols)

    if target_col in feature_cols:
        feature_cols.remove(target_col)

    if feature_cols and target_col:
        try:
            # Drop NA rows
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]

            # Train Linear Regression model
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            data["Predicted"] = predictions

            # Performance Metrics
            r2 = r2_score(y, predictions)
            mse = mean_squared_error(y, predictions)

            st.subheader("📊 Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("R² Score", f"{r2:.4f}", "✅ Good fit" if r2 > 0.7 else "⚠️ Poor fit")
            col2.metric("Mean Squared Error", f"{mse:,.2f}")

            # Scatter Plot: Actual vs Predicted
            st.subheader("📉 Prediction vs Actual Sales")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y, predictions, alpha=0.6, label="Predictions")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label="Ideal Fit")
            ax.set_xlabel("Actual Sales")
            ax.set_ylabel("Predicted Sales")
            ax.set_title("Actual vs Predicted Sales using Linear Regression")
            ax.legend()
            st.pyplot(fig)

            # Prediction Results Table
            st.subheader("📋 Prediction Results")
            preview_df = data.copy()
            preview_df["Error"] = preview_df["Predicted"] - preview_df[target_col]
            st.dataframe(preview_df[feature_cols + [target_col, "Predicted", "Error"]].round(2))

            # Custom input form
            st.subheader("🧪 Predict with Custom Input")
            with st.form("custom_input_form"):
                custom_values = {}
                cols = st.columns(len(feature_cols))
                for i, col in enumerate(feature_cols):
                    default_val = float(df[col].mean())
                    custom_values[col] = cols[i].number_input(f"{col}", value=default_val)
                submitted = st.form_submit_button("🚀 Predict")

                if submitted:
                    input_df = pd.DataFrame([custom_values])
                    prediction = model.predict(input_df)[0]
                    st.success(f"📌 Predicted {target_col}: {prediction:.2f}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("Please select valid feature(s) and a numeric target column.")
else:
    st.info("Upload a CSV file to begin.")
