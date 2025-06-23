import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set up page
st.set_page_config(page_title="🛒 Sales Prediction App", layout="wide")
st.title("🛒 Sales Prediction App")

# Sidebar: Upload CSV
with st.sidebar:
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.success("✅ Data uploaded!")

        # Model Metrics
        st.markdown("### 📈 Model Metrics")

# Main Panel
if uploaded_file:
    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # Feature + Target Selection
    st.subheader("🎯 Select Features and Target")
    feature_cols = st.multiselect("Select Feature Columns (numeric only):", numeric_cols)
    target_col = st.selectbox("Select Target Column:", [col for col in numeric_cols if col not in feature_cols])

    if feature_cols and target_col:
        # Drop NA
        data = df[feature_cols + [target_col]].dropna()
        X = data[feature_cols]
        y = data[target_col]

        # Train Model
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        data['Predicted'] = predictions

        # Metrics
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)

        # Show metrics in sidebar
        with st.sidebar:
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")

        # Highlight Predicted Column
        def highlight_pred(s):
            return ['background-color: #FFD700' if col == 'Predicted' else '' for col in s.index]

        st.subheader("📊 Prediction Results")
        st.dataframe(data.style.apply(highlight_pred, axis=1))

        # 📈 Chart
        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.regplot(x=y, y=predictions, line_kws={"color": "red"}, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Sales")
        st.pyplot(fig)

        # 📅 Custom Prediction
        st.subheader("🔮 Predict for Custom Input")
        custom_inputs = []
        cols = st.columns(len(feature_cols))
        for i, col in enumerate(feature_cols):
            val = cols[i].number_input(f"{col}", value=float(X[col].mean()))
            custom_inputs.append(val)

        if st.button("📌 Predict Custom Value"):
            custom_pred = model.predict([custom_inputs])[0]
            st.success(f"📈 Predicted Sales: {custom_pred:.2f}")

        # 📥 Download
        st.download_button("📥 Download Predictions as CSV", data.to_csv(index=False), file_name="predictions.csv")

    else:
        st.warning("⚠️ Please select at least one feature and a different target column.")
else:
    st.info("👆 Upload a CSV file to begin.")
