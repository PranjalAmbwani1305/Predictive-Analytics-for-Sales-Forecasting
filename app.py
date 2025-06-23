import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Streamlit Config
st.set_page_config(page_title="📊 Sales Forecasting App", layout="wide")
st.title("📈 Sales Forecasting with Linear Regression")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Filter numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    with st.form("config"):
        st.subheader("⚙️ Model Configuration")
        feature_cols = st.multiselect("✅ Select Feature Columns:", options=numeric_cols)
        target_col = st.selectbox("🎯 Select Target Column:", options=numeric_cols)
        future_periods = st.number_input("🔮 Predict how many future periods?", min_value=1, max_value=36, value=3)
        submit = st.form_submit_button("🚀 Run Model")

    if submit:
        if not feature_cols or target_col in feature_cols:
            st.error("❗ Please choose valid feature(s) and ensure target is not in features.")
        else:
            # Clean and split data
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            data['Predicted_' + target_col] = predictions

            # Metrics
            r2 = r2_score(y, predictions)
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)

            st.sidebar.markdown("### 📊 Model Metrics")
            st.sidebar.write(f"**R² Score:** {r2:.3f}")
            st.sidebar.write(f"**MAE:** {mae:.2f}")
            st.sidebar.write(f"**RMSE:** {rmse:.2f}")

            # Show Table
            st.subheader("🧾 Prediction Results")
            st.dataframe(data.style.highlight_max(axis=0, subset=['Predicted_' + target_col], color="lightgreen"))

            # Custom Prediction Form
            st.subheader("🧪 Predict for Custom Input")
            with st.form("custom"):
                custom_inputs = {}
                for col in feature_cols:
                    custom_inputs[col] = st.number_input(f"🔢 {col}", value=float(data[col].mean()))
                predict_button = st.form_submit_button("📍 Predict")

            if predict_button:
                input_df = pd.DataFrame([custom_inputs])
                custom_pred = model.predict(input_df)[0]
                st.success(f"✅ Predicted {target_col}: **{custom_pred:.2f}**")

            # Predict Future
            st.subheader("🔮 Predict Future (Trend)")
            last_known = data[feature_cols].iloc[-1].values.reshape(1, -1)
            future_preds = []
            future_inputs = last_known.copy()

            for _ in range(future_periods):
                next_pred = model.predict(future_inputs)[0]
                future_preds.append(next_pred)
                # Optionally modify future_inputs to simulate change

            future_df = pd.DataFrame({
                "Period": [f"Future {i+1}" for i in range(future_periods)],
                f"Predicted_{target_col}": future_preds
            })
            st.write(future_df)

            # Plot Actual vs Predicted
            st.subheader("📉 Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y.values, label="Actual", marker='o')
            ax.plot(predictions, label="Predicted", marker='x')
            ax.set_title("Actual vs Predicted Sales")
            ax.set_xlabel("Data Points")
            ax.set_ylabel(target_col)
            ax.legend()
            st.pyplot(fig)

            # Plot Future Trend
            st.subheader("📊 Future Trend Forecast")
            plt.figure(figsize=(10, 4))
            plt.plot(range(len(y)), y, label="Historical", marker='o')
            plt.plot(range(len(y), len(y) + future_periods), future_preds, label="Future Forecast", marker='x', linestyle="--")
            plt.title("Future Sales Forecast")
            plt.xlabel("Time Periods")
            plt.ylabel(target_col)
            plt.legend()
            st.pyplot(plt)

            # Download
            download_df = pd.concat([data, pd.DataFrame(future_preds, columns=[f"Predicted_{target_col}"])], axis=0)
            csv = download_df.to_csv(index=False)
            st.download_button("📥 Download Full Prediction CSV", csv, "predictions.csv", "text/csv")

else:
    st.info("Please upload a dataset to start.")
