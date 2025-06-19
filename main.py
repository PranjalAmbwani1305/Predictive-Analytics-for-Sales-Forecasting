import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("🛍️ Sales Prediction App")

# CSV Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    st.write("🧾 Available columns:")
    st.write(df.columns.tolist())

    # Let user select input features and target
    with st.form("select_columns"):
        st.subheader("🔧 Select Features and Target")
        feature_cols = st.multiselect("Select feature columns (independent variables):", df.columns)
        target_col = st.selectbox("Select target column (sales):", df.columns)
        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        if len(feature_cols) == 0:
            st.error("Please select at least one feature column.")
        elif target_col in feature_cols:
            st.error("Target column should not be selected as a feature.")
        else:
            try:
                # Prepare data
                X = df[feature_cols]
                y = df[target_col]

                # Train a simple model
                model = LinearRegression()
                model.fit(X, y)

                # Predict
                df['Predicted_' + target_col] = model.predict(X)

                st.subheader("📊 Prediction Results")
                st.write(df[feature_cols + [target_col, 'Predicted_' + target_col]])
            except Exception as e:
                st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to get started.")
