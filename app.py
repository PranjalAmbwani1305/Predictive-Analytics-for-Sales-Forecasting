import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="📊 Sales Predictor", layout="wide")
st.title("🛍️ Smart Sales Prediction App")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.error("❌ No numeric columns found.")
    else:
        with st.form("select_columns"):
            st.subheader("🎯 Select Features & Target")
            feature_cols = st.multiselect("✅ Feature Columns", options=numeric_cols)
            target_col = st.selectbox("🎯 Target Column", options=numeric_cols)
            submit = st.form_submit_button("🚀 Predict")

        if submit:
            if not feature_cols:
                st.error("⚠️ Please select at least one feature.")
            elif target_col in feature_cols:
                st.error("⚠️ Target column cannot be in feature list.")
            else:
                data = df[feature_cols + [target_col]].dropna()
                X = data[feature_cols]
                y = data[target_col]

                model = LinearRegression()
                model.fit(X, y)
                data["🔮 Predicted"] = model.predict(X)

                # Metrics
                r2 = r2_score(y, data["🔮 Predicted"])
                mae = mean_absolute_error(y, data["🔮 Predicted"])
                rmse = np.sqrt(mean_squared_error(y, data["🔮 Predicted"]))

                # Sidebar Metrics
                with st.sidebar:
                    st.header("📈 Model Performance")
                    st.metric("R²", f"{r2:.3f}")
                    st.metric("MAE", f"{mae:.2f}")
                    st.metric("RMSE", f"{rmse:.2f}")

                # Prediction Table
                st.subheader("📋 Prediction Results")
                styled_df = data.style.highlight_max(axis=0, subset=["🔮 Predicted"], color="lightgreen")
                st.dataframe(styled_df, use_container_width=True)

                # Custom Prediction
                st.subheader("🧠 Predict for Custom Input")
                input_data = []
                col1, col2 = st.columns(2)
                for i, col in enumerate(feature_cols):
                    with (col1 if i % 2 == 0 else col2):
                        val = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
                        input_data.append(val)

                if st.button("🎯 Predict Custom Input"):
                    input_df = pd.DataFrame([input_data], columns=feature_cols)
                    result = model.predict(input_df)[0]
                    st.success(f"🔮 Predicted {target_col}: **{result:.2f}**")

                # Graph
                st.subheader("📉 Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x=y, y=data["🔮 Predicted"], ax=ax, s=60, color='dodgerblue', label='Predictions')
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                ax.legend()
                st.pyplot(fig)

                # Download button
                st.download_button("📥 Download Results", data.to_csv(index=False), "predictions.csv")

else:
    st.info("Upload a CSV file to begin.")
