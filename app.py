import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    st.write("🧾 Available columns:")
    st.write(df.columns.tolist())

    # Filter numeric columns only
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        st.error("❌ Your file does not contain numeric columns for training.")
    else:
        with st.form("select_columns"):
            st.subheader("🔧 Select Features and Target")
            feature_cols = st.multiselect("✅ Select feature columns (numeric only):", numeric_cols)
            target_col = st.selectbox("🎯 Select target column:", numeric_cols)
            submitted = st.form_submit_button("🚀 Run Prediction")

        if submitted:
            if not feature_cols:
                st.error("Please select at least one feature column.")
            elif target_col in feature_cols:
                st.error("Target column cannot be one of the features.")
            else:
                try:
                    # Prepare data
                    X = df[feature_cols]
                    y = df[target_col]
                    data = pd.concat([X, y], axis=1).dropna()
                    X = data[feature_cols]
                    y = data[target_col]

                    # Train model
                    model = LinearRegression()
                    model.fit(X, y)

                    # Predict
                    predictions = model.predict(X)
                    data['Predicted_' + target_col] = predictions

                    # Calculate metrics
                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)
                    rmse = np.sqrt(mse)

                    # Fit interpretation
                    fit_quality_r2 = "Good Fit" if r2 >= 0.7 else "Moderate Fit" if r2 >= 0.3 else "Bad Fit"
                    fit_quality_mse = "Low Error" if rmse < 1000 else "Moderate Error" if rmse < 3000 else "High Error"

                    # Display metrics
                    st.subheader("📊 Evaluation Metrics")
                    st.markdown(f"📈 **R² Score**: `{r2:.4f}` ({fit_quality_r2})")
                    st.markdown(f"📉 **Mean Squared Error**: `{mse:,.2f}` (RMSE ≈ `{rmse:,.2f}`) → ({fit_quality_mse})")

                    # Display results
                    st.subheader("📄 Prediction Results")
                    st.write(data[feature_cols + [target_col, 'Predicted_' + target_col]])

                    # Visualization
                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(data[target_col], data['Predicted_' + target_col], alpha=0.6)
                    ax.plot([data[target_col].min(), data[target_col].max()],
                            [data[target_col].min(), data[target_col].max()],
                            color='red', linestyle='--')
                    ax.set_xlabel("Actual Sales")
                    ax.set_ylabel("Predicted Sales")
                    ax.set_title("Actual vs Predicted Sales")
                    st.pyplot(fig)

                    # Download option
                    st.download_button("📥 Download Predictions as CSV", data.to_csv(index=False), "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
