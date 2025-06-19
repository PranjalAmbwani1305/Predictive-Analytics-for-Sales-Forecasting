import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
from requests.exceptions import HTTPError

# Page setup
st.set_page_config(page_title="Retail Sales Prediction", page_icon="🛒", layout="wide")
st.title("🛍️ Retail Sales Prediction App")

# Load Hugging Face API token
HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", None)
if HF_TOKEN:
    qa_client = InferenceClient(token=HF_TOKEN)
else:
    qa_client = None
    st.sidebar.error("❗ Add `HUGGINGFACEHUB_API_TOKEN` in secrets for Q&A to work")

# Sidebar - Q&A
st.sidebar.header("❓ Ask a Question")
user_question = st.sidebar.text_input("About the model or dataset:")

if user_question:
    if qa_client:
        context = """
        This app uses Linear Regression to predict retail sales based on numeric features.
        R² Score measures how well predictions match the actual sales; closer to 1 is better.
        Mean Squared Error is the average squared difference between actual and predicted values (lower is better).
        """
        try:
            answer = qa_client.question_answering(
                question=user_question,
                context=context,
                model="deepset/roberta-base-squad2"
            )
            st.sidebar.success(f"💡 Answer: {answer['answer']}")
        except HTTPError as e:
            st.sidebar.error("❌ QA request failed. Please try a different question.")
    else:
        st.sidebar.info("🛠️ QA not available—missing API token.")

# Upload CSV and prediction logic
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric columns for modeling.")
    else:
        with st.form("model_form"):
            st.subheader("⚙️ Select Features & Target")
            feature_cols = st.multiselect("Features:", numeric_cols)
            target_col = st.selectbox("Target:", numeric_cols)
            submit = st.form_submit_button("🚀 Predict")

        if submit:
            if not feature_cols:
                st.error("Select at least one feature.")
            elif target_col in feature_cols:
                st.error("Target cannot be among features.")
            else:
                data = df[feature_cols + [target_col]].dropna()
                X, y = data[feature_cols], data[target_col]
                model = LinearRegression().fit(X, y)
                data["Predicted"] = model.predict(X)
                
                r2 = r2_score(y, data["Predicted"])
                mse = mean_squared_error(y, data["Predicted"])
                fit_label = (
                    "🟢 Excellent fit" if r2 > 0.75 else
                    "🟡 Moderate fit" if r2 > 0.40 else
                    "🔴 Poor fit"
                )

                st.success(f"📈 R² Score: {r2:.4f} ({fit_label})")
                st.info(f"📉 Mean Squared Error: {mse:,.2f}")

                st.subheader("📋 Prediction Results")
                st.dataframe(data, use_container_width=True)

                st.subheader("📈 Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(8,5))
                ax.scatter(y, data["Predicted"], alpha=0.6)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

                csv_data = data.to_csv(index=False)
                st.download_button("📥 Download Results", csv_data, "predictions.csv", "text/csv")
else:
    st.info("📁 Upload a CSV file to get started.")

st.markdown("---")
