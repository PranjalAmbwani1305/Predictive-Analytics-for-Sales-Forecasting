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
    future_periods = st.sidebar.slider("🔮 Predict Next N Periods", 1, 12, 3)

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
        st.dataframe(result_df.style.highlight_max(axis=0, subset=['Predicted'], color="lightgreen"))

        # 📈 Actual vs Predicted Plot
        st.subheader("📊 Actual vs Predicted Plot")
        fig1, ax1 = plt.subplots()
        ax1.plot(result_df[target_col].values, label='Actual', marker='o')
        ax1.plot(result_df['Predicted'].values, label='Predicted', marker='x')
        ax1.set_title("Actual vs Predicted")
        ax1.set_xlabel("Index")
        ax1.set_ylabel(target_col)
        ax1.legend()
        st.pyplot(fig1)

        # 🔮 Future Forecasting (Static assumption: use last known X)
        st.subheader("📅 Future Forecasting")
        last_input = X.iloc[-1:].copy()
        future_preds = []
        for _ in range(future_periods):
            next_pred = model.predict(last_input)[0]
            future_preds.append(next_pred)
            # You can update last_input dynamically if time-series data available

        future_df = pd.DataFrame({
            "Period": [f"Future {i+1}" for i in range(future_periods)],
            f"Predicted_{target_col}": future_preds
        })
        st.write(future_df)

        # 📉 Future Forecast Plot
        fig2, ax2 = plt.subplots()
        ax2.plot(range(len(y)), y, label='Actual', marker='o')
        ax2.plot(range(len(y)), predictions, label='Predicted', marker='x')
        ax2.plot(range(len(y), len(y)+future_periods), future_preds, label='Future Forecast', linestyle='--', marker='^')
        ax2.set_title("Full Sales Forecast")
        ax2.set_xlabel("Time Index")
        ax2.set_ylabel(target_col)
        ax2.legend()
        st.pyplot(fig2)

        # 📥 Download CSV
        full_df = pd.concat([result_df, pd.DataFrame({
            col: [np.nan]*future_periods for col in feature_cols
        }).assign(**{target_col: np.nan, "Predicted": future_preds})], ignore_index=True)
        st.download_button("📁 Download Predictions", full_df.to_csv(index=False), "predictions.csv", "text/csv")

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

    else:
        st.warning("Please select valid feature(s) and target column (target ≠ feature).")
else:
    st.info("👈 Upload a CSV to begin.")
