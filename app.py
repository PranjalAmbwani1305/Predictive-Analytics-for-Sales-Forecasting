import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit page config
st.set_page_config(page_title="📊 Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction Dashboard")

# File Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

# Process file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    with st.form("column_selection_form"):
        st.subheader("🔧 Select Features and Target")
        feature_cols = st.multiselect("✅ Select feature columns (numeric only):", numeric_cols)
        target_col = st.selectbox("🎯 Select target column:", numeric_cols)
        submitted = st.form_submit_button("🚀 Run Prediction")

    if submitted:
        if not feature_cols:
            st.error("❗ Please select at least one feature column.")
        elif target_col in feature_cols:
            st.error("❗ Target column cannot be one of the features.")
        else:
            try:
                data = df[feature_cols + [target_col]].dropna()
                X = data[feature_cols]
                y = data[target_col]

                # Train the model
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)
                data["Predicted"] = predictions

                # Metrics
                r2 = r2_score(y, predictions)
                mae = mean_absolute_error(y, predictions)
                mse = mean_squared_error(y, predictions)
                rmse = np.sqrt(mse)

                # Sidebar metrics
                st.sidebar.subheader("📊 Model Performance")
                st.sidebar.metric("R² Score", f"{r2:.4f}")
                st.sidebar.metric("MAE", f"{mae:.2f}")
                st.sidebar.metric("RMSE", f"{rmse:.2f}")

                # Prediction Results Table with Highlighted Column
                st.subheader("📋 Prediction Results")

                def highlight_predicted(s):
                    return ['background-color: lightgreen' if col == 'Predicted' else '' for col in s.index]

                styled_df = data.style.apply(lambda x: highlight_predicted(x), axis=1)
                st.dataframe(styled_df, use_container_width=True)

                # Graph
                st.subheader("📈 Actual vs Predicted")

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(y, predictions, color='dodgerblue', alpha=0.7, label="Predicted")
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit (y = x)')
                ax.set_xlabel("Actual Sales")
                ax.set_ylabel("Predicted Sales")
                ax.set_title("Linear Regression: Actual vs Predicted")
                ax.legend()
                st.pyplot(fig)

                # Custom prediction form
                st.subheader("🧪 Predict Custom Values")
                with st.form("custom_input_form"):
                    custom_inputs = {}
                    for col in feature_cols:
                        default_val = float(df[col].mean())
                        val = st.number_input(f"{col}:", value=default_val)
                        custom_inputs[col] = val
                    if st.form_submit_button("🔮 Predict"):
                        input_df = pd.DataFrame([custom_inputs])
                        result = model.predict(input_df)[0]
                        st.success(f"📌 Predicted {target_col}: {result:.2f}")

                # Download
                st.download_button("📥 Download Results as CSV", data.to_csv(index=False), "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
else:
    st.info("📤 Please upload a CSV file to get started.")
