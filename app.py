import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# App config
st.set_page_config(page_title="📈 Sales Forecast App", layout="wide")
st.title("📊 Sales Forecasting using Linear Regression")

# Upload file
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.sidebar.subheader("⚙️ Configuration")
    feature_cols = st.sidebar.multiselect("Select Feature Columns", numeric_cols)
    target_col = st.sidebar.selectbox("Select Target Column", numeric_cols)

    if feature_cols and target_col and target_col not in feature_cols:
        # Drop missing
        clean_df = df[feature_cols + [target_col]].dropna()
        X = clean_df[feature_cols]
        y = clean_df[target_col]

        # Train model
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # Metrics
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)

        st.sidebar.markdown("### 📈 Model Performance")
        st.sidebar.write(f"**R²:** {r2:.3f}")
        st.sidebar.write(f"**MAE:** {mae:.2f}")
        st.sidebar.write(f"**RMSE:** {rmse:.2f}")

        # Show results
        result_df = clean_df.copy()
        result_df['Predicted'] = predictions
        st.subheader("🧾 Actual vs Predicted Table")
        st.dataframe(result_df)  # No highlighting

        # 📊 Actual vs Predicted Scatter Plot
        st.subheader("📊 Actual vs Predicted Sales")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(result_df[target_col], result_df['Predicted'], edgecolors='black', alpha=0.7, label='Predicted Points')
        ax.plot([result_df[target_col].min(), result_df[target_col].max()],
                [result_df[target_col].min(), result_df[target_col].max()],
                'r--', label='Ideal Line (y = x)')
        ax.set_xlabel("Actual Sales", fontsize=12)
        ax.set_ylabel("Predicted Sales", fontsize=12)
        ax.set_title("Actual vs Predicted Sales", fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

        # 🧪 Custom Prediction
        st.subheader("🎯 Predict for Custom Values")
        with st.form("custom_form"):
            custom_vals = {}
            for col in feature_cols:
                val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
                custom_vals[col] = val
            predict_btn = st.form_submit_button("📍 Predict Now")
        if predict_btn:
            input_df = pd.DataFrame([custom_vals])
            custom_result = model.predict(input_df)[0]
            st.success(f"🧾 Predicted {target_col}: **{custom_result:.2f}**")

        # 📥 Download CSV
        st.download_button("📁 Download Predictions", result_df.to_csv(index=False), "predictions.csv", "text/csv")

    else:
        st.warning("Please select valid feature(s) and target column (target ≠ feature).")
else:
    st.info("👈 Upload a CSV to begin.")
