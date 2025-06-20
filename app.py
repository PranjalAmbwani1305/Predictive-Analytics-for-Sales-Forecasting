import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from transformers import pipeline

# ----------------------- SETUP -----------------------
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# ------------------- LOAD QA MODEL --------------------
@st.cache_resource
def load_qa_pipeline():
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2", token=st.secrets["HUGGINGFACE_TOKEN"])
    except:
        return None

qa_pipeline = load_qa_pipeline()

# -------------------- SIDEBAR QA ----------------------
st.sidebar.header("💡 Dataset / Model Queries")
question = st.sidebar.text_input("Ask a question about the model or dataset:")

if question:
    if qa_pipeline is None:
        st.sidebar.warning("⚠️ QA system not available (check Hugging Face token or internet connection).")
    else:
        context = """
        R² Score (Coefficient of Determination) measures how well predictions approximate actual values.
        RMSE (Root Mean Squared Error) is the square root of the average of squared differences between predicted and actual values.
        In Linear Regression, the model finds the best linear relationship between selected features and the target.
        """
        result = qa_pipeline(question=question, context=context)
        st.sidebar.success("💡 Answer: " + result['answer'])

# ------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    st.write("🧾 Available Columns:")
    st.write(df.columns.tolist())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    with st.form("select_columns"):
        st.subheader("🔧 Select Features and Target")
        feature_cols = st.multiselect("Select feature columns:", df.columns.tolist())
        target_col = st.selectbox("Select target column:", numeric_cols)
        submitted = st.form_submit_button("🚀 Run Prediction")

    if submitted:
        if not feature_cols:
            st.error("Please select at least one feature column.")
        elif target_col in feature_cols:
            st.error("Target column cannot be one of the features.")
        elif df[target_col].dtype == 'object':
            st.error("Target column must be numeric.")
        else:
            try:
                data = df[feature_cols + [target_col]].dropna()
                X = data[feature_cols]
                y = data[target_col]

                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)

                data['Predicted_' + target_col] = predictions
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                rmse = mean_squared_error(y, predictions, squared=False)

                st.success("✅ Prediction completed!")

                # Summary
                st.markdown(f"""
                ### 📈 Model Performance
                - **R² Score:** {r2:.4f} ({"Good fit" if r2 > 0.7 else "Poor fit"})
                - **RMSE:** {rmse:.2f}
                - **MSE:** {mse:.2f}
                """)

                # Results
                st.subheader("📊 Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🧠 Features & Target**")
                    st.dataframe(data[feature_cols + [target_col]])
                with col2:
                    st.markdown("**🎯 Predictions**")
                    st.dataframe(data[['Predicted_' + target_col]])

                # Graph
                st.subheader("📉 Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(y, predictions, alpha=0.6)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted Values")
                st.pyplot(fig)

                # Download
                st.download_button("📥 Download Results", data.to_csv(index=False), "predictions.csv", "text/csv")

                # Custom Input Prediction
                st.subheader("🧪 Predict Custom Values")
                with st.form("custom_input"):
                    inputs = {}
                    for col in feature_cols:
                        value = st.text_input(f"{col}:", value=str(df[col].mean()))
                        try:
                            inputs[col] = float(value)
                        except:
                            st.error(f"Invalid value for {col}.")
                    predict_btn = st.form_submit_button("Predict Custom Values")

                if predict_btn:
                    input_df = pd.DataFrame([inputs])
                    custom_pred = model.predict(input_df)[0]
                    st.success(f"📌 Predicted {target_col}: {custom_pred:.2f}")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

else:
    st.info("Upload a CSV file to get started.")

# ----------------- REMOVE FOOTER -------------------
st.markdown("""
    <style>
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
