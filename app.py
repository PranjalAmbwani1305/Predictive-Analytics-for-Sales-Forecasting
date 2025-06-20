import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="📈 Sales Prediction Dashboard", layout="wide")
st.title("📊 Sales Prediction with Linear Regression")

# Fix dropdown/text visibility for dark theme
st.markdown("""
<style>
    div[data-baseweb="select"] > div {
        color: white !important;
    }
    .stSelectbox>div>div>div>div {
        color: white !important;
    }
    input, textarea {
        background-color: #262730 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Upload CSV
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
        run = st.form_submit_button("🚀 Run Prediction")

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

                # Plot
                st.markdown("### 📉 Actual vs Predicted Plot")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y, y=preds, label="Predictions", ax=ax)
                sns.lineplot(x=y, y=y, color="red", linestyle="--", label="Ideal Fit", ax=ax)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Sales")
                ax.legend()
                st.pyplot(fig)

                # Download button
                st.download_button("📥 Download Predictions", result_df.to_csv(index=False), "predictions.csv", "text/csv")

                # Custom input prediction
                st.markdown("### 🧪 Predict on Custom Input")
                with st.form("custom_input_form"):
                    custom_vals = {}
                    for col in features:
                        val = st.text_input(f"Enter value for **{col}**:", value=str(df[col].iloc[0]))
                        custom_vals[col] = val
                    predict_btn = st.form_submit_button("🎯 Predict")

                    if predict_btn:
                        try:
                            custom_df = pd.DataFrame([custom_vals])
                            # Convert numerical if needed
                            for col in custom_df.columns:
                                try:
                                    custom_df[col] = pd.to_numeric(custom_df[col])
                                except:
                                    pass
                            custom_df_encoded = pd.get_dummies(custom_df)
                            # Align with training features
                            missing_cols = set(X.columns) - set(custom_df_encoded.columns)
                            for col in missing_cols:
                                custom_df_encoded[col] = 0
                            custom_df_encoded = custom_df_encoded[X.columns]
                            prediction = model.predict(custom_df_encoded)[0]
                            st.success(f"📌 Predicted {target}: `{prediction:.2f}`")
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
else:
    st.info("📤 Please upload a CSV file to get started.")
