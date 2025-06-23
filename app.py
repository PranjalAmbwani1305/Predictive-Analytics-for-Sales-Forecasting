import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🔮 Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Numeric columns only
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric columns found.")
    else:
        with st.form("selection_form"):
            st.subheader("🔧 Select Features and Target")
            feature_cols = st.multiselect("✅ Feature columns (numeric only):", options=numeric_cols)
            target_col = st.selectbox("🎯 Target column:", options=numeric_cols)
            submitted = st.form_submit_button("🚀 Train & Predict")

        if submitted:
            if not feature_cols:
                st.error("Please select feature columns.")
            elif target_col in feature_cols:
                st.error("Target column must not be among features.")
            else:
                try:
                    # Clean and prepare data
                    data = df[feature_cols + [target_col]].dropna()
                    X = data[feature_cols]
                    y = data[target_col]

                    # Train model
                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(X)

                    # Add predictions
                    data['Predicted_' + target_col] = predictions

                    # Metrics in sidebar
                    r2 = r2_score(y, predictions)
                    mae = mean_absolute_error(y, predictions)
                    mse = mean_squared_error(y, predictions)
                    rmse = np.sqrt(mse)

                    st.sidebar.markdown("### 📊 Model Metrics")
                    st.sidebar.metric("R² Score", f"{r2:.3f}")
                    st.sidebar.metric("MAE", f"{mae:.2f}")
                    st.sidebar.metric("RMSE", f"{rmse:.2f}")

                    # Table with highlight
                    st.subheader("📊 Prediction Results Table")
                    st.dataframe(
                        data.style.highlight_max(axis=0, subset=['Predicted_' + target_col], color='lightgreen')
                    )

                    # Plot Actual vs Predicted
                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(x=y, y=predictions, ax=ax, color='blue', edgecolor='black')
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

                    # Custom input prediction
                    st.subheader("✍️ Predict with Custom Input")
                    with st.form("custom_input_form"):
                        custom_inputs = {}
                        for col in feature_cols:
                            default_val = float(X[col].mean())
                            custom_inputs[col] = st.number_input(f"{col}", value=default_val)
                        custom_submit = st.form_submit_button("Predict")
                    if custom_submit:
                        input_df = pd.DataFrame([custom_inputs])
                        custom_prediction = model.predict(input_df)[0]
                        st.success(f"📢 Predicted {target_col}: **{custom_prediction:.2f}**")

                    # Download CSV
                    st.download_button(
                        "📥 Download Results",
                        data.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
else:
    st.info("📤 Please upload a CSV file to get started.")
