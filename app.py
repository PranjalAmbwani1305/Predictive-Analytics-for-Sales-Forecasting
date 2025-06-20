import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="📈 Sales Prediction Dashboard", layout="wide")

# 🌙 Dark theme styling
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .st-bw, .st-cj, .st-bo { color: white !important; }
        .stSelectbox>div>div>div>div { color: black !important; }
        .css-1v0mbdj p { color: white; }
        .css-1y4p8pa { color: white; }
        .stButton button { background-color: #4CAF50; color: white; }
        .stTextInput>div>div>input { background-color: #262730; color: white; }
        .stDataFrame { background-color: #262730; }
        .css-1rs6os.edgvbvh3 { color: black !important; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛠️ Filters")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])

# Title
st.title("📊 Sales Forecasting Dashboard")
st.write("Upload your dataset and select features to forecast sales using a linear regression model.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown("---")
    st.subheader("🔧 Feature & Target Selection")

    col1, col2 = st.columns(2)
    with col1:
        feature_cols = st.multiselect("✅ Select feature columns (numeric only):", options=numeric_cols)
    with col2:
        target_col = st.selectbox("🎯 Select target column (numeric only):", options=numeric_cols)

    # Prevent selecting the same column as both feature and target
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    if feature_cols and target_col:
        try:
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            data["Predicted"] = predictions

            r2 = r2_score(y, predictions)
            mse = mean_squared_error(y, predictions)

            # Metrics
            st.subheader("📈 Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("R² Score", f"{r2:.4f}", "✅ Good fit" if r2 > 0.7 else "⚠️ Needs improvement")
            col2.metric("Mean Squared Error", f"{mse:,.2f}")

            # Plotting Actual vs Predicted
            st.subheader("📊 Prediction vs Actual Plot")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y, predictions, alpha=0.7, label="Predictions")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label="Ideal")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            st.pyplot(fig)

            # Results Table
            st.subheader("📋 Prediction Results Table")
            st.dataframe(data[feature_cols + [target_col, "Predicted"]].style.format(precision=2))

            # Custom Prediction Input
            st.subheader("🧪 Predict on Custom Input")
            with st.form("custom_input_form"):
                custom_values = {}
                for col in feature_cols:
                    val = st.number_input(f"Enter value for {col}:", value=float(df[col].mean()))
                    custom_values[col] = val
                submitted = st.form_submit_button("🚀 Predict")
                if submitted:
                    input_df = pd.DataFrame([custom_values])
                    prediction = model.predict(input_df)[0]
                    st.success(f"📌 Predicted {target_col}: {prediction:.2f}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.warning("Please select valid numeric feature(s) and a target column.")

else:
    st.info("Please upload a CSV file to begin.")

# Hide footer & hamburger
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
