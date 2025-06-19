import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from transformers import pipeline
import math

# Load QA Pipeline
@st.cache_resource
def load_qa_pipeline():
    try:
        return pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            token=st.secrets["HUGGINGFACE_TOKEN"]
        )
    except Exception as e:
        return None

qa_pipeline = load_qa_pipeline()
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# App UI setup
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# Sidebar - Queries section
st.sidebar.header("🧠 Ask a Dataset Question")
if qa_pipeline:
    user_question = st.sidebar.text_input("Type your question:")
    context = """
    R² Score (Coefficient of Determination) measures how well predictions match actual values.
    RMSE (Root Mean Squared Error) shows the average difference between predicted and actual values.
    Lower RMSE and higher R² means better fit.
    Linear Regression is used to predict continuous values based on features.
    """
    if user_question:
        try:
            with st.spinner("💬 Answering..."):
                answer = qa_pipeline(question=user_question, context=context)
                st.sidebar.markdown(f"💡 **Answer:** {answer['answer']}")
        except Exception as e:
            st.sidebar.error("❌ QA request failed. Please try again.")
else:
    st.sidebar.warning("QA system not available (check Hugging Face token or internet connection).")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    st.write("🧾 Available Columns:")
    st.write(df.columns.tolist())

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
                st.error("❌ Please select at least one feature column.")
            elif target_col in feature_cols:
                st.error("❌ Target column cannot be one of the features.")
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

                    st.success("✅ Prediction completed!")
                    st.subheader("📊 Prediction Results")
                    st.write(data[feature_cols + [target_col, 'Predicted_' + target_col]])

                    # Display metrics
                    st.subheader("📈 Model Performance")
                    st.markdown(f"**R² Score**: {r2:.4f} ({'Good fit' if r2 > 0.7 else 'Moderate fit' if r2 > 0.4 else 'Poor fit'})")
                    st.markdown(f"**RMSE**: {rmse:.2f}")
                    st.markdown(f"**MSE**: {mse:.2f}")

                    # Visualization
                    st.subheader("📉 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(y, predictions, alpha=0.6)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
                    ax.set_xlabel("Actual Sales")
                    ax.set_ylabel("Predicted Sales")
                    ax.set_title("Actual vs Predicted Sales")
                    st.pyplot(fig)

                    # Download button
                    st.download_button(
                        "📥 Download Predictions as CSV",
                        data.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
else:
    st.info("📤 Please upload a CSV file to begin.")
