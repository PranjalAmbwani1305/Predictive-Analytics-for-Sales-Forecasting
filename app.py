import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from io import StringIO

# App config
st.set_page_config(page_title="📈 Sales Forecast App", layout="wide")
st.title("📊 Sales Forecasting")

# Upload file
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("The uploaded file must contain at least two numeric columns.")
        st.stop()

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
        st.dataframe(result_df)

        # 📊 Actual vs Predicted Scatter Plot
        st.subheader("📊 Actual vs Predicted Sales")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(result_df[target_col], result_df['Predicted'],
                   edgecolors='black', alpha=0.7, label='Predicted Points')
        lims = [min(result_df[target_col].min(), result_df['Predicted'].min()),
                max(result_df[target_col].max(), result_df['Predicted'].max())]
        ax.plot(lims, lims, 'r--', label='Ideal Line (y = x)')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
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

        # 📥 Download Predictions CSV
        st.download_button(
            "📁 Download Predictions",
            result_df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

        # 📄 Summary Report Download
        st.subheader("📄 Download Summary Report")
        summary = StringIO()
        summary.write("📊 Sales Forecasting Summary Report\n")
        summary.write("="*40 + "\n\n")
        summary.write("🗂 Selected Features:\n")
        summary.write(", ".join(feature_cols) + "\n\n")
        summary.write(f"🎯 Target Column:\n{target_col}\n\n")
        summary.write("📈 Model Performance:\n")
        summary.write(f"- R² Score: {r2:.3f}\n")
        summary.write(f"- Mean Absolute Error (MAE): {mae:.2f}\n")
        summary.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}\n\n")
        summary.write(f"📅 Rows used for training: {len(clean_df)}\n")
        summary.write(f"📁 Original file name: {uploaded_file.name}\n")

        st.download_button(
            label="📄 Download Summary Report",
            data=summary.getvalue(),
            file_name="sales_forecast_summary.txt",
            mime="text/plain"
        )

    else:
        st.warning("Please select valid feature(s) and target column (target ≠ feature).")
else:
    st.info("👈 Upload a CSV to begin.")
