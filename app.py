import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from transformers import pipeline

# ----------------- PAGE CONFIG ----------------- #
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.markdown("""<style>.st-emotion-cache-1dp5vir {visibility: hidden;}</style>""", unsafe_allow_html=True)  # Remove footer

st.title("🛍️ Sales Prediction App")

# ----------------- LOAD QA MODEL ----------------- #
@st.cache_resource
def load_qa_pipeline():
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2", token=st.secrets["HUGGINGFACE_TOKEN"])
    except Exception as e:
        st.warning("⚠️ QA system not available (check Hugging Face token or internet connection).")
        return None

qa_pipeline = load_qa_pipeline()

# ----------------- SIDEBAR: QUERIES ----------------- #
st.sidebar.title("❓ Dataset & Model Queries")
question = st.sidebar.text_input("Ask a question about the dataset/model:")
if st.sidebar.button("🔎 Ask"):
    if qa_pipeline and question:
        context = """
        R² Score (R-squared) measures how well the predicted values from the model match the actual values.
        Mean Squared Error (MSE) is the average of the squares of the errors between actual and predicted values.
        RMSE is the square root of MSE.
        Linear Regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables.
        """
        try:
            answer = qa_pipeline(question=question, context=context)
            st.sidebar.write(f"💡 Answer: {answer['answer']}")
        except:
            st.sidebar.error("❌ QA request failed. Try again.")
    else:
        st.sidebar.info("Enter a question to get started.")

# ----------------- MAIN APP ----------------- #
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

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

                    # Append predictions
                    data['Predicted_' + target_col] = predictions

                    # Evaluation
                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)
                    rmse = mean_squared_error(y, predictions, squared=False)

                    # Interpretation
                    fit_quality = "🟢 Good Fit" if r2 > 0.7 else ("🟡 Moderate Fit" if r2 > 0.4 else "🔴 Poor Fit")

                    st.success("✅ Prediction completed!")
                    st.subheader("📊 Prediction Results")
                    st.write(data[feature_cols + [target_col, 'Predicted_' + target_col]])

                    st.subheader("📉 Model Evaluation")
                    st.write(f"📈 R² Score: {r2:.4f} ({fit_quality})")
                    st.write(f"📉 Mean Squared Error: {mse:.2f}")
                    st.write(f"📐 RMSE: {rmse:.2f}")

                    # Plot
                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(y, predictions, alpha=0.6)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
                    ax.set_xlabel("Actual Sales")
                    ax.set_ylabel("Predicted Sales")
                    ax.set_title("Actual vs Predicted Sales")
                    st.pyplot(fig)

                    # Download
                    st.download_button("📥 Download Predictions as CSV", data.to_csv(index=False), "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")
