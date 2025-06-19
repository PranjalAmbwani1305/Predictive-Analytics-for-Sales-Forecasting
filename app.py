import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from transformers import pipeline
import os

# --------------------------
# Optional: Remove Streamlit footer
hide_footer_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

# --------------------------
# Page config
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# --------------------------
# Load QA model (Hugging Face)
@st.cache_resource
def load_qa_pipeline():
    try:
        return pipeline("question-answering",
                        model="deepset/roberta-base-squad2",
                        token=st.secrets["HUGGINGFACE_TOKEN"])
    except Exception as e:
        return None

qa_pipeline = load_qa_pipeline()

# --------------------------
# Sidebar: QA
st.sidebar.title("💡 Dataset / Model Queries")
question = st.sidebar.text_area("Ask a question about the model or dataset:")
answer_placeholder = st.sidebar.empty()

if question:
    if qa_pipeline is not None:
        context_text = """
        R² Score (Coefficient of Determination) measures how well the predictions approximate the actual values.
        MSE (Mean Squared Error) is the average of the squared differences between predicted and actual values.
        RMSE is the square root of MSE.
        Linear Regression fits a line by minimizing MSE.
        """
        try:
            result = qa_pipeline(question=question, context=context_text)
            answer_placeholder.markdown(f"💡 **Answer:** {result['answer']}")
        except:
            answer_placeholder.warning("⚠️ QA request failed. Try again.")
    else:
        answer_placeholder.warning("⚠️ QA system not available (check Hugging Face token or internet connection).")

# --------------------------
# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    st.write("🧾 Available columns:")
    st.write(df.columns.tolist())

    # Numeric columns only
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

                    # Metrics
                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)

                    # Fit Quality
                    fit_quality = "🟢 Good Fit" if r2 > 0.7 else ("🟡 Moderate Fit" if r2 > 0.3 else "🔴 Poor Fit")

                    # Display Results
                    st.success("✅ Prediction completed!")
                    st.subheader("📊 Prediction Results")
                    st.write(data[feature_cols + [target_col, 'Predicted_' + target_col]])

                    # Metrics Display
                    st.subheader("📌 Model Evaluation")
                    st.write(f"📈 R² Score: {r2:.4f} ({fit_quality})")
                    st.write(f"📉 Mean Squared Error: {mse:,.2f}")

                    # Plot
                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(y, predictions, alpha=0.6)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

                    # Download
                    st.download_button("📥 Download Predictions as CSV",
                                       data.to_csv(index=False),
                                       "predictions.csv",
                                       "text/csv")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

else:
    st.info("📤 Please upload a CSV file to begin.")
