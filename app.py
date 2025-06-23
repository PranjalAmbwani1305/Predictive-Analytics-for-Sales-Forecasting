import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Sales Forecasting App", layout="wide")

st.title("📈 Sales Forecasting App with Linear Regression")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        st.error("❌ The dataset must contain at least two numeric columns.")
    else:
        with st.form("column_selection"):
            st.subheader("🧩 Select Features and Target")
            feature_cols = st.multiselect("✅ Feature columns (independent variables):", numeric_cols)
            target_col = st.selectbox("🎯 Target column (dependent variable):", numeric_cols)
            submit_btn = st.form_submit_button("🚀 Run Prediction")

        if submit_btn:
            if not feature_cols or target_col in feature_cols:
                st.error("⚠️ Please select valid features and a distinct target.")
            else:
                try:
                    data = df[feature_cols + [target_col]].dropna()
                    X = data[feature_cols]
                    y = data[target_col]

                    model = LinearRegression()
                    model.fit(X, y)

                    predictions = model.predict(X)
                    data['Predicted_' + target_col] = predictions

                    # Sidebar Metrics
                    st.sidebar.subheader("📉 Model Metrics")
                    r2 = r2_score(y, predictions)
                    mae = mean_absolute_error(y, predictions)
                    rmse = np.sqrt(mean_squared_error(y, predictions))

                    st.sidebar.metric("R² Score", f"{r2:.4f}")
                    st.sidebar.metric("MAE", f"{mae:.2f}")
                    st.sidebar.metric("RMSE", f"{rmse:.2f}")

                    # Show Prediction Table
                    st.subheader("📊 Prediction Results")
                    styled_table = data.style.highlight_max(axis=0, subset=['Predicted_' + target_col], color='lightgreen')
                    st.dataframe(styled_table)

                    # Graph: Actual vs Predicted
                    st.subheader("📉 Actual vs Predicted Line Plot")
                    fig, ax = plt.subplots()
                    ax.plot(y.values, label="Actual", marker='o')
                    ax.plot(predictions, label="Predicted", marker='x')
                    ax.set_title("Actual vs Predicted Values")
                    ax.set_xlabel("Data Points")
                    ax.set_ylabel(target_col)
                    ax.legend()
                    st.pyplot(fig)

                    # Future Prediction Section
                    st.subheader("🔮 Predict Future Sales")
                    future_periods = st.number_input("📅 Number of future periods to predict", min_value=1, max_value=12, value=3)
                    future_inputs = []

                    st.write("🔢 Enter values for each feature for future periods:")

                    for i in range(future_periods):
                        st.markdown(f"**Period {i+1}**")
                        row = []
                        for col in feature_cols:
                            row.append(st.number_input(f"{col} (Period {i+1})", key=f"{col}_{i}"))
                        future_inputs.append(row)

                    if st.button("Predict Future"):
                        try:
                            future_df = pd.DataFrame(future_inputs, columns=feature_cols)
                            future_preds = model.predict(future_df)
                            future_df['Predicted_' + target_col] = future_preds
                            st.success("✅ Future predictions done.")
                            st.dataframe(future_df)

                            # Future plot
                            fig2, ax2 = plt.subplots()
                            ax2.plot(future_preds, label='Future Prediction', marker='o', linestyle='--', color='purple')
                            ax2.set_title("🔮 Future Sales Forecast")
                            ax2.set_xlabel("Future Period")
                            ax2.set_ylabel(target_col)
                            ax2.legend()
                            st.pyplot(fig2)
                        except Exception as e:
                            st.error(f"❌ Error predicting future values: {e}")

                    # Custom Input Prediction
                    st.subheader("🧪 Predict for Custom Input")
                    with st.form("custom_input_form"):
                        custom_vals = []
                        for col in feature_cols:
                            val = st.number_input(f"{col} (custom)", value=float(X[col].mean()), key=f"custom_{col}")
                            custom_vals.append(val)
                        custom_submit = st.form_submit_button("🔍 Predict")

                    if custom_submit:
                        try:
                            input_df = pd.DataFrame([custom_vals], columns=feature_cols)
                            pred = model.predict(input_df)[0]
                            st.success(f"✅ Predicted {target_col}: **{pred:.2f}**")
                        except Exception as e:
                            st.error(f"❌ Custom input prediction error: {e}")

                    # Download button
                    st.download_button(
                        label="📥 Download Prediction Results",
                        data=data.to_csv(index=False),
                        file_name="sales_predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Processing error: {e}")

else:
    st.info("📂 Please upload a CSV file to begin.")
