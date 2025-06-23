import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📈 Sales Prediction Dashboard", layout="wide")

st.title("📊 Sales Forecasting App")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview Data")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    with st.form("feature_target_selection"):
        st.subheader("🎯 Select Features and Target")
        feature_cols = st.multiselect("✅ Select feature columns:", numeric_cols)
        target_col = st.selectbox("🎯 Select target column:", numeric_cols)
        submit_btn = st.form_submit_button("🚀 Run Prediction")

    if submit_btn:
        if not feature_cols:
            st.warning("⚠️ Please select at least one feature column.")
        elif target_col in feature_cols:
            st.error("❌ Target column cannot be one of the feature columns.")
        else:
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]

            # Model training
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            # Metrics
            r2 = r2_score(y, predictions)
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)

            # Display metrics
            with st.sidebar:
                st.markdown("### 📈 Model Performance")
                st.metric("R²", f"{r2:.3f}")
                st.metric("MAE", f"{mae:.2f}")
                st.metric("RMSE", f"{rmse:.2f}")

            # Store predictions
            data["🔮 Predicted"] = predictions

            # Graph: Actual vs Predicted
            st.subheader("📉 Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(x=y, y=predictions, ax=ax, line_kws={"color": "red"}, scatter_kws={"alpha": 0.6})
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("📈 Linear Regression Fit")
            st.pyplot(fig)

            # Highlight prediction column
            styled_table = data.style.highlight_max(axis=0, subset=["🔮 Predicted"], color="lightgreen")
            st.subheader("📋 Prediction Results")
            st.dataframe(styled_table, use_container_width=True)

            # Custom prediction form
            st.subheader("✏️ Predict for Custom Input")
            with st.expander("🔍 Enter custom values"):
                input_values = {}
                for col in feature_cols:
                    default_val = float(df[col].dropna().iloc[0])
                    input_values[col] = st.number_input(f"{col}", value=default_val)

                if st.button("🧠 Predict Now"):
                    input_df = pd.DataFrame([input_values])
                    prediction = model.predict(input_df)[0]
                    st.success(f"🔮 Predicted {target_col}: **{prediction:.2f}**")

            # Download button
            st.download_button("📥 Download Predictions", data.to_csv(index=False), file_name="predictions.csv")

else:
    st.info("📤 Please upload a CSV file to start.")
