import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers.pipelines import PipelineException

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Retail Sales Prediction App", page_icon="🛒", layout="wide")
st.title("🛍️ Retail Sales Prediction App")

# -------------------- QA Pipeline Setup --------------------
@st.cache_resource
def load_qa_pipeline():
    try:
        return pipeline("question-answering", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")
    except Exception as e:
        st.sidebar.warning("⚠️ QA model unavailable. Running in offline mode.")
        return None

qa_pipeline = load_qa_pipeline()

# -------------------- Sidebar: Dataset Queries --------------------
st.sidebar.header("❓ Ask Questions About Your Data")
user_query = st.sidebar.text_input("Type your question:")

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    st.write("🧾 Available columns:")
    st.write(df.columns.tolist())

    # -------------------- Show QA Answer (if question and model available) --------------------
    if user_query and qa_pipeline:
        try:
            answer = qa_pipeline(question=user_query, context=df.head(20).to_string())
            st.sidebar.success(f"💡 Answer: {answer['answer']}")
        except Exception:
            st.sidebar.error("Unable to process the question. Try a simpler query.")
    elif user_query and not qa_pipeline:
        st.sidebar.info("📴 QA system not available in offline mode.")

    # -------------------- Prediction Section --------------------
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
                    y_pred = model.predict(X)
                    data['Predicted_' + target_col] = y_pred

                    # Evaluation Metrics
                    r2 = r2_score(y, y_pred)
                    mse = mean_squared_error(y, y_pred)

                    # Fit quality interpretation
                    if r2 >= 0.75:
                        fit_quality = "Excellent Fit"
                    elif r2 >= 0.50:
                        fit_quality = "Good Fit"
                    elif r2 >= 0.25:
                        fit_quality = "Fair Fit"
                    else:
                        fit_quality = "Poor Fit"

                    # Display results
                    st.success("✅ Prediction completed!")
                    st.subheader("📊 Prediction Results")
                    st.write(data[feature_cols + [target_col, 'Predicted_' + target_col]])

                    # Display metrics
                    st.markdown(f"📈 **R² Score**: `{r2:.4f}` ({fit_quality})")
                    st.markdown(f"📉 **Mean Squared Error**: `{mse:,.2f}`")

                    # Plotting
                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(y, y_pred, alpha=0.6)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
                    ax.set_xlabel("Actual Sales")
                    ax.set_ylabel("Predicted Sales")
                    ax.set_title("Actual vs Predicted Sales")
                    st.pyplot(fig)

                    # Download
                    st.download_button("📥 Download Predictions as CSV",
                                       data.to_csv(index=False),
                                       "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

else:
    st.info("📎 Please upload a CSV file to begin.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("🔗 [GitHub Repository](https://github.com/KOdoi-OJ/CF_Time_Series_Forecasting_Project) | Made with ❤️ by *KME*")
