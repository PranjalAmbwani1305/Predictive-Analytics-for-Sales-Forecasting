import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import base64

# ----------------- Page Configuration -------------------
st.set_page_config(page_title="📈 Sales Prediction Pro", layout="wide")

# ----------------- Custom CSS Styling -------------------
st.markdown("""
    <style>
        body { background-color: #f9f9f9; font-family: 'Segoe UI', sans-serif; }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1.5em;
            margin-top: 0.5em;
            font-size: 1rem;
        }
        .stDownloadButton button {
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1.5em;
            margin-top: 1em;
        }
        .stSelectbox label, .stMultiselect label {
            font-weight: bold;
            font-size: 1rem;
        }
        .stDataFrame { border-radius: 10px; overflow: hidden; }
    </style>
""", unsafe_allow_html=True)

# ----------------- Title -------------------
st.title("🛍️ Sales Prediction Dashboard")
st.markdown("Upload your dataset, select features, and forecast sales using a Linear Regression model.")

# ----------------- File Upload -------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric columns found for training.")
    else:
        with st.form("input_form"):
            st.markdown("### 🔧 Model Configuration")
            col1, col2 = st.columns(2)

            with col1:
                feature_cols = st.multiselect("✅ Select feature columns:", numeric_cols, default=numeric_cols[:-1])
            with col2:
                target_col = st.selectbox("🎯 Select target column:", numeric_cols, index=len(numeric_cols) - 1)

            submitted = st.form_submit_button("🚀 Run Prediction")

        if submitted:
            if not feature_cols:
                st.error("Please select at least one feature column.")
            elif target_col in feature_cols:
                st.error("Target column cannot also be a feature.")
            else:
                try:
                    data = df[feature_cols + [target_col]].dropna()
                    X = data[feature_cols]
                    y = data[target_col]

                    model = LinearRegression()
                    model.fit(X, y)
                    data["Predicted"] = model.predict(X)

                    st.success("✅ Prediction completed!")

                    # ----------- Metrics -----------
                    st.markdown("### 📊 Model Insights")
                    r_squared = model.score(X, y)
                    st.metric("R² Score", f"{r_squared:.3f}", delta="Good" if r_squared > 0.8 else "Needs Improvement")

                    # ----------- Result Table -----------
                    with st.expander("📋 Show Prediction Table"):
                        st.dataframe(data[feature_cols + [target_col, "Predicted"]], use_container_width=True)

                    # ----------- Visualization -----------
                    st.markdown("### 📈 Actual vs Predicted")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.set(style="whitegrid")
                    sns.scatterplot(x=data[target_col], y=data["Predicted"], color='royalblue', s=70, ax=ax, label='Predicted')
                    sns.regplot(x=data[target_col], y=data["Predicted"], scatter=False, color='orange', ax=ax, label='Trend Line')
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')

                    ax.set_xlabel("Actual Sales")
                    ax.set_ylabel("Predicted Sales")
                    ax.set_title("Actual vs Predicted Sales")
                    ax.legend()
                    st.pyplot(fig)

                    # ----------- Download Button -----------
                    st.download_button(
                        label="📥 Download Predictions",
                        data=data.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
else:
    st.info("⬆️ Upload a CSV file to start.")
