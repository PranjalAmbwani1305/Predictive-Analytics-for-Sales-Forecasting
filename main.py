import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    st.write("🧾 Available columns:")
    st.write(df.columns.tolist())

    # Select features and target
    with st.form("select_columns"):
        st.subheader("🔧 Select Features and Target")
        feature_cols = st.multiselect("✅ Select feature columns:", df.columns)
        target_col = st.selectbox("🎯 Select target column:", df.columns)
        submitted = st.form_submit_button("🚀 Run Prediction")

    if submitted:
        if not feature_cols:
            st.error("Please select at least one feature column.")
        elif target_col in feature_cols:
            st.error("Target column cannot be in features.")
        else:
            try:
                # Prepare and clean data
                X = df[feature_cols]
                y = df[target_col]

                data = pd.concat([X, y], axis=1).dropna()
                X = data[feature_cols]
                y = data[target_col]

                # Train model
                model = LinearRegression()
                model.fit(X, y)

                # Predict
                data['Predicted_' + target_col] = model.predict(X)

                st.success("✅ Prediction completed!")
                st.subheader("📊 Prediction Results")
                st.write(data[feature_cols + [target_col, 'Predicted_' + target_col]])

                st.download_button("📥 Download Predictions as CSV", data.to_csv(index=False), "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
