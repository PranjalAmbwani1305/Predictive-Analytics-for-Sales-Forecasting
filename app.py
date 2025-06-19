import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient

# ----------- Page Setup -----------
st.set_page_config(page_title="Retail Sales Prediction", page_icon="🛒", layout="wide")
st.title("🛍️ Retail Sales Prediction App")

# ----------- Hugging Face Inference Setup -----------

# Make sure to set your token in an environment variable: HUGGINGFACEHUB_API_TOKEN
HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", None)
if not HF_TOKEN:
    st.sidebar.error("❗ Hugging Face API token missing. Add HUGGINGFACEHUB_API_TOKEN to Streamlit secrets.")
    qa_client = None
else:
    qa_client = InferenceClient(token=HF_TOKEN)

# ----------- Sidebar: Q&A -----------

st.sidebar.header("❓ Ask a Question")
user_query = st.sidebar.text_input("About the model or dataset:")

if user_query:
    if qa_client:
        context = """
        This app uses Linear Regression to predict retail sales based on numeric features.
        R² Score tells how well predictions match actual values: 1.0 = perfect.
        Mean Squared Error is the average squared prediction error (lower is better).
        """
        try:
            result = qa_client.text_generation(
                user_query,
                model="deepset/roberta-base-squad2",
                max_new_tokens=150
            )
            answer = result[0]["generated_text"]
            st.sidebar.success(f"💡 Answer: {answer}")
        except Exception as e:
            st.sidebar.error("❌ QA request failed. Try again.")
    else:
        st.sidebar.info("📴 QA is offline—please check your API token.")

# ----------- Main App: Upload and Predict -----------

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric columns found.")
    else:
        with st.form("model_form"):
            st.subheader("⚙️ Choose inputs")
            features = st.multiselect("Features:", numeric_cols)
            target = st.selectbox("Target:", numeric_cols)
            submit = st.form_submit_button("🚀 Predict")

        if submit:
            if not features:
                st.error("Select at least one feature.")
            elif target in features:
                st.error("Target can't be one of the features.")
            else:
                data = df[features + [target]].dropna()
                X, y = data[features], data[target]
                model = LinearRegression().fit(X, y)
                data["Predicted"] = model.predict(X)

                r2 = r2_score(y, data["Predicted"])
                mse = mean_squared_error(y, data["Predicted"])
                fit_msg = (
                    "🟢 Excellent fit" if r2 >= 0.75 else
                    "🟡 Moderate fit" if r2 >= 0.5 else
                    "🔴 Poor fit"
                )

                st.success(f"📈 R²: {r2:.4f} ({fit_msg})")
                st.info(f"📉 MSE: {mse:,.2f}")

                st.subheader("📋 Results")
                st.dataframe(data, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(y, data["Predicted"], alpha=0.6)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
                ax.set(xlabel="Actual", ylabel="Predicted")
                st.pyplot(fig)

                csv = data.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")
else:
    st.info("📁 Upload a CSV to begin.")

# ----------- Footer -----------
st.markdown("---")
st.markdown("🔗 [Repo](https://github.com/KOdoi-OJ/CF_Time_Series_Forecasting_Project) &bull; Made with ❤️ by KME")
