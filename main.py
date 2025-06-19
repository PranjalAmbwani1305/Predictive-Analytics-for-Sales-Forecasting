import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Sales Prediction App")

# CSV Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(df)

    # Simple check
    if 'feature1' in df.columns and 'sales' in df.columns:
        X = df[['feature1']]  # Example: change as per your features
        y = df['sales']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        df['predicted_sales'] = model.predict(X)

        st.subheader("Predicted Sales")
        st.write(df[['feature1', 'sales', 'predicted_sales']])
    else:
        st.error("CSV must contain 'feature1' and 'sales' columns for this demo.")
