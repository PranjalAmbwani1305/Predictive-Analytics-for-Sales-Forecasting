import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from transformers import pipeline
import math

# ------------------- Setup -------------------
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction & Analysis App")

# Load QA pipeline with Hugging Face token (ensure it's available in secrets)
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/roberta-base-squad2", 
                    token=st.secrets["HUGGINGFACE_TOKEN"])

qa_pipeline = load_qa_pipeline()

# ------------------- Sidebar -------------------
st.sidebar.title("🔍 Dataset Queries")

user_question = st.sidebar.text_input("Ask a question about the model, metrics, or dataset:")
if user_question:
    context = """
    This app uses Linear Regression to predict sales based on numeric features.

    📊 Metrics Explained:
    - R² Score (Coefficient of Determination): Measures how well predictions match actual values. Closer to 1 means better fit.
    - MSE (Mean Squared Error): Average of squared differences between predicted and actual values.
    - RMSE (Root Mean Squared Error): Square root of MSE. RMSE = sqrt(MSE). It’s in the same unit as the target variable and is easier to interpret.

    🧠 Example:
    If MSE is 100, then RMSE = sqrt(100) = 10.
    """
    try:
        response = qa_pipeline(question=user_question, context=context)
        st.sidebar.success(f"💡 Answer: {response['answer']}")
    except Exception as e:
        st.sidebar.error("❌ QA request failed. Try again.")

# ------------------- Main Interface -------------------

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("❌ Your file does not contain numeric columns for training.")
    else:
        with st.form("column_selection"):
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
                    X = df[feature_cols]
                    y = df[target_col]
                    data = pd.concat([X, y], axis=1).dropna()
                    X = data[feature_cols]
                    y = data[target_col]

                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(X)

                    data['Predicted_' + target_col] = predictions

                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)
                    rmse = math.sqrt(mse)

                    fit_quality = "🔺 Good Fit" if r2 > 0.6 else ("⚠️ Moderate Fit" if r2 > 0.3 else "❌ Poor Fit")

                    st.success("✅ Prediction completed!")
                    st.subheader("📊 Prediction Results")
                    st.write(data[feature_cols + [target_col, 'Predicted_' + target_col]])

                    st.subheader("🔢 Model Metrics")
                    st.markdown(f"- **R² Score**: {r2:.4f} ({fit_quality})")
                    st.markdown(f"- **Mean Squared Error (MSE)**: {mse:,.2f}")
                    st.markdown(f"- **Root Mean Squared Error (RMSE)**: {rmse:,.2f}")

                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(data[target_col], data['Predicted_' + target_col], alpha=0.6)
                    ax.plot([data[target_col].min(), data[target_col].max()],
                            [data[target_col].min(), data[target_col].max()],
                            color='red', linestyle='--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

                    st.download_button("📥 Download Predictions as CSV",
                                       data.to_csv(index=False),
                                       "predictions.csv",
                                       "text/csv")
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
else:
    st.info("Upload a CSV file to begin.")
