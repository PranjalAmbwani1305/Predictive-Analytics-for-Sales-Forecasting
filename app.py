import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from transformers import pipeline
import numpy as np

# Set page
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# Remove footer
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Hugging Face QA Setup ---
@st.cache_resource
def load_qa_pipeline():
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2",
                        token=st.secrets["HUGGINGFACE_TOKEN"])
    except Exception:
        return None

qa_pipeline = load_qa_pipeline()

# --- Sidebar QA chatbot ---
with st.sidebar:
    st.subheader("💬 Dataset / Model Queries")
    user_question = st.text_input("Ask a question about the model or dataset:")
    if user_question:
        if qa_pipeline is not None:
            context_text = """
            This app uses Linear Regression to predict sales based on numeric features.
            R² Score measures how well predictions match actual results.
            RMSE (Root Mean Squared Error) shows the average prediction error.
            Prediction is made using scikit-learn's LinearRegression().
            """
            try:
                answer = qa_pipeline(question=user_question, context=context_text)
                st.success(f"💡 Answer: {answer['answer']}")
            except:
                st.warning("⚠️ QA request failed. Try again.")
        else:
            st.warning("⚠️ QA system not available (check Hugging Face token or internet connection).")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    st.write("🧾 Available numeric columns:")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [col for col in numeric_cols if "year" not in col.lower()]
    st.write(numeric_cols)

    if not numeric_cols:
        st.error("❌ No valid numeric columns found for prediction.")
    else:
        with st.form("select_columns"):
            st.subheader("🔧 Select Features and Target")
            feature_cols = st.multiselect("✅ Select feature columns:", numeric_cols)
            target_options = [col for col in numeric_cols if col not in feature_cols]
            target_col = st.selectbox("🎯 Select target column:", target_options)
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
                    data["Predicted"] = predictions

                    # Evaluation metrics
                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)
                    rmse = np.sqrt(mse)

                    st.success("✅ Prediction completed!")
                    st.markdown(f"📈 **R² Score**: {r2:.4f} {'(Good Fit)' if r2 > 0.75 else '(Bad Fit)' if r2 < 0.4 else '(Moderate Fit)'}")
                    st.markdown(f"📉 **RMSE**: {rmse:.2f}")

                    # Results Table
                    st.subheader("📊 Prediction Results")
                    st.dataframe(data[feature_cols + [target_col, "Predicted"]], use_container_width=True)

                    # Improved Visualization
                    st.subheader("📈 Actual vs Predicted Plot")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(y, predictions, alpha=0.7, label="Predicted vs Actual")
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Perfect Fit")
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    ax.legend()
                    st.pyplot(fig)

                    # Download option
                    st.download_button("📥 Download Predictions", data.to_csv(index=False), "predictions.csv", "text/csv")

                    # --- Custom Input Prediction ---
                    st.subheader("🧮 Predict Custom Values")
                    with st.form("custom_input_form"):
                        input_data = {}
                        for col in feature_cols:
                            input_data[col] = st.number_input(f"Enter value for {col}:", value=float(df[col].mean()))
                        predict_btn = st.form_submit_button("🔍 Predict")

                    if predict_btn:
                        input_df = pd.DataFrame([input_data])
                        custom_prediction = model.predict(input_df)[0]
                        st.success(f"📌 Predicted {target_col}: **{custom_prediction:.2f}**")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
else:
    st.info("📂 Please upload a CSV file to begin.")
