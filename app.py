import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="📈 Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction Dashboard")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    with st.form("feature_form"):
        st.subheader("🔧 Select Features and Target")
        col1, col2 = st.columns(2)
        feature_cols = col1.multiselect("✅ Select Feature Columns:", options=numeric_cols)
        target_col = col2.selectbox("🎯 Select Target Column:", options=numeric_cols)
        submitted = st.form_submit_button("🚀 Train Model")

    if submitted:
        if target_col in feature_cols:
            st.error("⚠️ Target column cannot be one of the feature columns.")
        elif not feature_cols:
            st.error("⚠️ Please select at least one feature column.")
        else:
            try:
                # Clean data
                data = df[feature_cols + [target_col]].dropna()
                X = data[feature_cols]
                y = data[target_col]

                # Train model
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)
                data["🔮 Predicted"] = predictions

                # Metrics
                r2 = r2_score(y, predictions)
                mae = mean_absolute_error(y, predictions)
                rmse = np.sqrt(mean_squared_error(y, predictions))

                # Show metrics in sidebar
                st.sidebar.subheader("📊 Model Performance")
                st.sidebar.metric("R² Score", f"{r2:.4f}")
                st.sidebar.metric("MAE", f"{mae:.2f}")
                st.sidebar.metric("RMSE", f"{rmse:.2f}")

                # Show result table with highlighted predicted column
                st.subheader("📋 Prediction Results")
                styled_table = data.style.highlight_max(axis=0, subset=["🔮 Predicted"], color='lightgreen')
                st.dataframe(styled_table, use_container_width=True)

                # Graph
                st.subheader("📈 Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(y, predictions, alpha=0.6, color='skyblue', edgecolor='black', label='Predictions')
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                ax.legend()
                st.pyplot(fig)

                # Custom input
                st.subheader("🧪 Predict for Custom Values")
                custom_vals = {}
                col_input = st.columns(len(feature_cols))
                for i, col in enumerate(feature_cols):
                    custom_vals[col] = col_input[i].number_input(f"{col}", value=float(X[col].mean()))
                if st.button("🔮 Predict Custom Sales"):
                    user_df = pd.DataFrame([custom_vals])
                    user_pred = model.predict(user_df)[0]
                    st.success(f"📌 Predicted {target_col}: **{user_pred:.2f}**")

            except Exception as e:
                st.error(f"❌ Error during model training: {e}")
else:
    st.info("Please upload a CSV file to begin.")
