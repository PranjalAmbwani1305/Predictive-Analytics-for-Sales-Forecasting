import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from transformers import pipeline
import os

st.set_page_config(page_title="📈 Sales Prediction Dashboard", layout="wide")

# Set dark theme-style appearance
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .css-1d391kg { background-color: #262730; }
        .css-1lcbmhc, .st-bw, .st-cj { color: white !important; }
        .stSelectbox>div>div>div>div { color: black !important; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛠️ Filters")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])

# Hugging Face QA Pipeline
@st.cache_resource
def load_qa_pipeline():
    try:
        from transformers import pipeline
        token = st.secrets["HUGGINGFACE_TOKEN"]
        return pipeline("question-answering", model="deepset/roberta-base-squad2", token=token)
    except:
        return None

qa_pipeline = load_qa_pipeline()

# Main Title
st.title("📊 Sales Forecasting Dashboard")
st.write("Upload your dataset and select features to forecast sales using a linear regression model.")

# Process CSV
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Uploaded Data")
    st.dataframe(df.head())

    with st.sidebar:
        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        feature_cols = st.multiselect("Select feature columns:", options=all_columns)
        target_col = st.selectbox("Select target column:", options=numeric_cols)

    # Remove target from feature list if included
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    # Run prediction
    if feature_cols and target_col:
        try:
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            data["Predicted"] = predictions

            r2 = r2_score(y, predictions)
            mse = mean_squared_error(y, predictions)

            # Metrics
            st.subheader("📈 Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("R² Score", f"{r2:.4f}", "✅ Good fit" if r2 > 0.7 else "⚠️ Poor fit")
            col2.metric("MSE", f"{mse:,.2f}")

            # Plot
            st.subheader("📊 Prediction vs Actual")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y, predictions, alpha=0.7)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted Sales")
            st.pyplot(fig)

            # Result Table
            st.subheader("📋 Prediction Results")
            st.dataframe(data[feature_cols + [target_col, "Predicted"]])

            # Custom Prediction Input
            with st.expander("🧪 Predict Custom Values"):
                custom_input = {}
                for col in feature_cols:
                    val = st.text_input(f"Enter value for {col}:", value=str(df[col].iloc[0]))
                    try:
                        val = float(val)
                    except:
                        val = 0.0
                    custom_input[col] = val

                if st.button("🚀 Predict Custom Values"):
                    input_df = pd.DataFrame([custom_input])
                    result = model.predict(input_df)[0]
                    st.success(f"📌 Predicted {target_col}: {result:.2f}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.warning("Please select valid features and a numeric target column.")

    # QA SYSTEM
    st.sidebar.title("💬 Dataset / Model Queries")
    question = st.sidebar.text_input("Ask a question about the model or dataset:")

    if question:
        if qa_pipeline:
            context = " ".join([f"{col}: {str(df[col].iloc[0])}" for col in df.columns])
            try:
                answer = qa_pipeline(question=question, context=context)
                st.sidebar.markdown(f"💡 **Answer:** {answer['answer']}")
            except Exception:
                st.sidebar.warning("⚠️ QA request failed. Try again.")
        else:
            st.sidebar.warning("⚠️ QA system not available (check Hugging Face token or internet connection).")

else:
    st.info("Please upload a CSV file to begin.")

# Remove Streamlit footer and hamburger
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
