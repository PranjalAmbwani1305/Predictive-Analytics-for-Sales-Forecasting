import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Retail Sales Prediction", page_icon=":bar_chart:", layout="wide")
st.title("📊 Retail Sales Prediction App")

# Load QA model
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased")

qa_pipeline = load_qa_pipeline()

# Sidebar Info
st.sidebar.header("📚 Column Info")
st.sidebar.markdown("""
- **Numerical columns** like oil_price, onpromotion, etc., are used for prediction.
- You can select which columns to use as features and what to predict.
""")

# Queries Section in Sidebar
st.sidebar.header("❓ Ask a Query")
user_question = st.sidebar.text_input("Ask something about the model or dataset:")

if user_question:
    context = """
    This app uses Linear Regression to predict retail sales. R² Score measures how well predictions match actual values;
    it ranges from 0 to 1. A value closer to 1 indicates a better fit. MSE (Mean Squared Error) shows average squared 
    difference between actual and predicted values (lower is better).
    """
    response = qa_pipeline(question=user_question, context=context)
    st.sidebar.success(f"**Answer:** {response['answer']}")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        st.error("❌ Your file does not contain numeric columns for training.")
    else:
        with st.form("select_columns"):
            st.subheader("⚙️ Select Features and Target")
            feature_cols = st.multiselect("✅ Choose feature columns:", numeric_cols)
            target_col = st.selectbox("🎯 Choose target column:", numeric_cols)
            submitted = st.form_submit_button("🚀 Run Prediction")

        if submitted:
            if not feature_cols:
                st.error("❗ Please select at least one feature column.")
            elif target_col in feature_cols:
                st.error("❗ Target column cannot be a feature.")
            else:
                try:
                    # Drop missing values
                    data = df[feature_cols + [target_col]].dropna()

                    # Train model
                    X = data[feature_cols]
                    y = data[target_col]
                    model = LinearRegression()
                    model.fit(X, y)

                    # Predict
                    predictions = model.predict(X)
                    data['Predicted_' + target_col] = predictions

                    # Metrics
                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)

                    fit_quality = "🔴 Poor fit"
                    if r2 > 0.7:
                        fit_quality = "🟢 Excellent fit"
                    elif r2 > 0.4:
                        fit_quality = "🟡 Moderate fit"

                    # Show results
                    st.success(f"📈 R² Score: {r2:.4f} ({fit_quality})")
                    st.info(f"📉 Mean Squared Error: {mse:,.2f}")

                    st.subheader("📋 Prediction Results")
                    st.dataframe(data, use_container_width=True)

                    # Plot
                    st.subheader("📉 Actual vs Predicted")
                    fig, ax = plt.subplots()
                    ax.scatter(y, predictions, alpha=0.6)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

                    # Download button
                    st.download_button(
                        label="📥 Download Results",
                        data=data.to_csv(index=False),
                        file_name="predicted_sales.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
else:
    st.info("📁 Upload a CSV file to start.")
