import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="📈 Sales Forecasting Dashboard", layout="wide")

# Dark theme styles
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .css-1d391kg { background-color: #262730; }
        .css-1lcbmhc, .st-bw, .st-cj { color: white !important; }
        .stSelectbox>div>div>div>div { color: black !important; }
        #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar: File Upload
st.sidebar.title("🛠️ Upload & Configure")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])

# App Header
st.title("📊 Sales Forecasting with Linear Regression")
st.write("Upload your dataset, select feature and target columns, and forecast sales with visual insights.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Sidebar Config
    with st.sidebar:
        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        feature_cols = st.multiselect("Select feature columns:", options=all_columns)
        target_col = st.selectbox("Select target column:", options=numeric_cols)

    # Avoid selecting target in features
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    if feature_cols and target_col:
        try:
            data = df[feature_cols + [target_col]].dropna()
            X = pd.get_dummies(data[feature_cols], drop_first=True)
            y = data[target_col]

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            data["Predicted"] = predictions

            r2 = r2_score(y, predictions)
            mse = mean_squared_error(y, predictions)

            # Performance Metrics
            st.subheader("📈 Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("R² Score", f"{r2:.4f}", "✅ Good fit" if r2 > 0.7 else "⚠️ Poor fit")
            col2.metric("Mean Squared Error", f"{mse:,.2f}")

            # Improved Prediction vs Actual Plot
            st.subheader("📊 Prediction vs Actual Sales")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.regplot(x=y, y=predictions, ci=None, line_kws={"color": "red"}, scatter_kws={'alpha':0.6})
            ax.set_xlabel("Actual Sales")
            ax.set_ylabel("Predicted Sales")
            ax.set_title("Actual vs Predicted Sales using Linear Regression")
            ax.grid(True)
            st.pyplot(fig)

            # Styled Prediction Table
            st.subheader("📋 Prediction Results")
            styled_data = data[feature_cols + [target_col, "Predicted"]].copy()
            st.dataframe(
                styled_data.style.format("{:.2f}", subset=[target_col, "Predicted"]),
                use_container_width=True
            )

            # Better Custom Input Form
            st.subheader("🧪 Predict with Custom Input")
            with st.form("custom_input_form"):
                custom_values = {}
                cols = st.columns(min(3, len(feature_cols)))  # layout in columns
                for idx, col in enumerate(feature_cols):
                    with cols[idx % len(cols)]:
                        val = st.text_input(f"{col}", value=str(df[col].iloc[0]))
                        try:
                            val = float(val)
                        except:
                            val = 0.0
                        custom_values[col] = val

                submit = st.form_submit_button("🚀 Predict Custom Value")

            if submit:
                input_df = pd.DataFrame([custom_values])
                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=X.columns, fill_value=0)
                prediction = model.predict(input_df)[0]
                st.success(f"📌 Predicted {target_col}: **{prediction:.2f}**")

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please select valid feature(s) and a numeric target column.")
else:
    st.info("📤 Upload a CSV file from the sidebar to start.")
