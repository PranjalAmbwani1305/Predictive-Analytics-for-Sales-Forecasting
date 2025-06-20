import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="📈 Sales Forecasting App", layout="wide")

# Custom Style
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .stSelectbox label, .stTextInput label, .stMultiselect label {
            color: #ffffff !important;
            font-weight: bold;
        }
        .stMetric { font-size: 1.2em; }
        .reportview-container .main footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Sales Forecasting Dashboard")
st.write("Upload your dataset, choose features, and predict sales using Linear Regression.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown("---")
    st.subheader("🔧 Feature & Target Selection")

    col1, col2 = st.columns(2)
    with col1:
        feature_cols = st.multiselect("✅ Select feature columns:", options=all_columns)
    with col2:
        target_col = st.selectbox("🎯 Select target column (numeric only):", options=numeric_cols)

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

            st.markdown("---")
            st.subheader("📊 Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("R² Score", f"{r2:.4f}", "✅ Good fit" if r2 > 0.7 else "⚠️ Poor fit")
            col2.metric("Mean Squared Error", f"{mse:,.2f}")

            st.markdown("---")
            st.subheader("📈 Actual vs Predicted Sales")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y, predictions, alpha=0.7, label="Predicted vs Actual")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Perfect Fit")
            ax.set_xlabel("Actual Sales")
            ax.set_ylabel("Predicted Sales")
            ax.set_title("Actual vs Predicted Comparison")
            ax.legend()
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("📋 Prediction Results Table")
            styled_table = data[feature_cols + [target_col, "Predicted"]].style.format("{:.2f}", subset=["Predicted"])
            st.dataframe(styled_table, use_container_width=True)

            # Custom Input
            st.markdown("---")
            with st.expander("🧪 Predict Custom Input"):
                st.markdown("Enter values below to predict a custom sale output.")
                custom_inputs = {}
                for col in feature_cols:
                    default_val = df[col].iloc[0] if pd.api.types.is_numeric_dtype(df[col]) else ""
                    value = st.text_input(f"{col}:", value=str(default_val))
                    try:
                        value = float(value)
                    except:
                        value = 0.0
                    custom_inputs[col] = value

                if st.button("🚀 Predict Custom Values"):
                    input_df = pd.DataFrame([custom_inputs])
                    result = model.predict(input_df)[0]
                    st.success(f"📌 Predicted {target_col}: **{result:.2f}**")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    else:
        st.warning("⚠️ Please select at least one feature and a numeric target column.")

else:
    st.info("📤 Upload a CSV file to get started.")
