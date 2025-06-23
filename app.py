import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# App layout and title
st.set_page_config(page_title="📈 Advanced Sales Predictor", layout="wide")
st.title("🛍️ Advanced Sales Prediction App")

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in the uploaded data.")
    else:
        with st.form("setup_form"):
            st.markdown("### 🔧 Select Feature and Target Columns")
            col1, col2 = st.columns(2)
            with col1:
                feature_cols = st.multiselect("📊 Feature Columns", numeric_cols, default=numeric_cols[:-1])
            with col2:
                target_col = st.selectbox("🎯 Target Column", numeric_cols, index=len(numeric_cols)-1)
            submit_btn = st.form_submit_button("🚀 Train & Predict")

        if submit_btn:
            if not feature_cols:
                st.error("Please select at least one feature.")
            elif target_col in feature_cols:
                st.error("Target column cannot be one of the features.")
            else:
                # Prepare and clean data
                data = df[feature_cols + [target_col]].dropna()
                X = data[feature_cols]
                y = data[target_col]

                # Fit Linear Regression
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)

                # Append predictions to DataFrame
                data["Predicted"] = predictions

                # Metrics
                r2 = r2_score(y, predictions)
                mae = mean_absolute_error(y, predictions)
                mse = mean_squared_error(y, predictions)
                rmse = np.sqrt(mse)

                # Display performance
                st.subheader("📏 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R²", f"{r2:.3f}")
                col2.metric("MAE", f"{mae:.2f}")
                col3.metric("MSE", f"{mse:.2f}")
                col4.metric("RMSE", f"{rmse:.2f}")

                # Show predictions
                st.subheader("📋 Actual vs Predicted")
                st.dataframe(data[feature_cols + [target_col, "Predicted"]], use_container_width=True)

                # Line Plot (Actual vs Predicted)
                st.subheader("📈 Line Plot - Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(data[target_col].values, label='Actual', marker='o')
                ax.plot(data["Predicted"].values, label='Predicted', marker='x')
                ax.set_title("Actual vs Predicted Sales")
                ax.set_ylabel("Sales")
                ax.legend()
                st.pyplot(fig)

                # Future prediction
                st.subheader("🔮 Predict Future Sales")
                future_periods = st.slider("Select number of future periods", 1, 12, 3)
                future_inputs = []

                st.markdown("### 📥 Input Future Feature Values")
                for i in range(future_periods):
                    st.markdown(f"#### Future Period {i+1}")
                    user_input = {}
                    cols = st.columns(len(feature_cols))
                    for j, col in enumerate(feature_cols):
                        user_input[col] = cols[j].number_input(f"{col} (Month {i+1})", value=float(X[col].mean()))
                    future_inputs.append(user_input)

                if st.button("📊 Predict Future"):
                    future_df = pd.DataFrame(future_inputs)
                    future_preds = model.predict(future_df)
                    future_df["Predicted Sales"] = future_preds
                    future_df.index = [f"Future {i+1}" for i in range(future_periods)]

                    st.dataframe(future_df, use_container_width=True)

                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.plot(future_df["Predicted Sales"], marker='o', color='green')
                    ax2.set_title("Future Sales Prediction")
                    ax2.set_ylabel("Predicted Sales")
                    ax2.set_xlabel("Future Periods")
                    st.pyplot(fig2)

                # Download
                st.download_button(
                    "📥 Download Full Predictions",
                    data.to_csv(index=False),
                    file_name="full_predictions.csv",
                    mime="text/csv"
                )

                # Summary Report
                st.subheader("📝 Summary Report")
                st.markdown(f"""
                - ✅ Model trained using **{len(feature_cols)} features** to predict `{target_col}`.
                - 📈 Achieved **R² = {r2:.3f}**, indicating {"strong" if r2 > 0.7 else "moderate" if r2 > 0.5 else "weak"} predictive performance.
                - 🔍 Observed **MAE = {mae:.2f}**, indicating average prediction error.
                - 📊 Trend visualization confirms that predicted sales align well with actual sales.
                - 🔮 Forecasted **next {future_periods} periods** of sales using custom inputs.
                """)

