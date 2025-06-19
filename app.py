
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Retail Sales Predictor", page_icon="📊", layout="wide")
st.title("🛍️ Retail Sales Prediction App")


with st.sidebar:
    st.header("ℹ️ Column Guidelines")
    st.markdown("""
    - Select **only numeric** columns for features and target.
    - Avoid using the **target column as a feature**.
    - File should be a clean CSV with no major missing data.
    - You can scale data before training if needed.
    """)

# ----- Upload File
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

# ----- Session to Store Results
if "results" not in st.session_state:
    st.session_state["results"] = []

# ----- Main Logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric columns found. Please upload a proper dataset.")
    else:
        with st.form("model_form"):
            st.subheader("🔧 Configure Model Inputs")

            feature_cols = st.multiselect("✅ Select feature columns:", numeric_cols)
            target_col = st.selectbox("🎯 Select target column:", numeric_cols)

            scale_option = st.checkbox("⚙️ Scale features (StandardScaler)", value=False)
            submitted = st.form_submit_button("🚀 Predict")

        if submitted:
            if not feature_cols:
                st.error("❗ Please select at least one feature column.")
            elif target_col in feature_cols:
                st.error("❗ Target column cannot be a feature.")
            else:
                # Drop NA rows
                data = df[feature_cols + [target_col]].dropna()
                if data.empty:
                    st.warning("⚠️ No data left after removing rows with missing values.")
                else:
                    X = data[feature_cols]
                    y = data[target_col]

                    if scale_option:
                        scaler = StandardScaler()
                        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

                    # Fit model
                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(X)

                    # Metrics
                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)

                    # Store results
                    result_df = data.copy()
                    result_df['Predicted_' + target_col] = predictions
                    st.session_state["results"].append(result_df)

                    # Display
                    st.success("✅ Prediction complete!")
                    st.subheader("📊 Prediction Results")
                    st.dataframe(result_df, use_container_width=True)

                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(y, predictions, alpha=0.6)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

                    st.write(f"📈 **R² Score**: `{r2:.4f}`")
                    st.write(f"📉 **Mean Squared Error**: `{mse:.2f}`")

                    # Download button
                    csv_data = result_df.to_csv(index=False)
                    st.download_button("📥 Download Predictions as CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")

    # Display previous predictions if available
    if st.session_state["results"]:
        with st.expander("📁 View Previous Prediction Runs"):
            all_results = pd.concat(st.session_state["results"], ignore_index=True)
            st.dataframe(all_results.tail(10), use_container_width=True)

else:
    st.info("⬆️ Upload a CSV file to begin.")
