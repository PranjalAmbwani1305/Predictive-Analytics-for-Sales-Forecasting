import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📈 Sales Prediction Dashboard", layout="wide")
st.title("📊 Sales Prediction with Linear Regression")

# File upload
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    with st.form("selection_form"):
        st.subheader("🔧 Select Features and Target")
        features = st.multiselect("Select Feature Columns:", df.columns.tolist())
        target = st.selectbox("Select Target Column:", [col for col in numeric_cols if col not in features])
        run = st.form_submit_button("🚀 Predict")

    if run:
        if not features:
            st.error("❗ Please select at least one feature.")
        else:
            try:
                data = df[features + [target]].dropna()
                X = pd.get_dummies(data[features], drop_first=True)
                y = data[target]

                model = LinearRegression()
                model.fit(X, y)
                preds = model.predict(X)

                # Evaluation
                r2 = r2_score(y, preds)
                mse = mean_squared_error(y, preds)
                rmse = np.sqrt(mse)

                # Results table
                result_df = data.copy()
                result_df["Predicted"] = preds
                result_df["Error"] = result_df["Predicted"] - result_df[target]
                result_df["Absolute Error"] = result_df["Error"].abs()
                result_df["% Error"] = 100 * result_df["Absolute Error"] / result_df[target]

                st.markdown("### ✅ Model Performance")
                st.markdown(f"""
                - **R² Score**: `{r2:.4f}` {"(Good fit ✅)" if r2 > 0.6 else "(Needs improvement ⚠️)"}
                - **MSE**: `{mse:,.2f}`
                - **RMSE**: `{rmse:,.2f}`
                """)

                st.markdown("### 🧾 Prediction Results")
                st.dataframe(result_df[[target, "Predicted", "Absolute Error", "% Error"]].round(2), use_container_width=True)

                # Better visual
                st.markdown("### 📉 Actual vs Predicted Plot")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y, y=preds, label="Predictions", ax=ax)
                sns.lineplot(x=y, y=y, color="red", linestyle="--", label="Ideal Fit", ax=ax)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Sales")
                ax.legend()
                st.pyplot(fig)

                st.download_button("📥 Download Predictions", result_df.to_csv(index=False), "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload a CSV file to begin.")
