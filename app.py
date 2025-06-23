import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# File uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric columns found.")
    else:
        with st.form("selection_form"):
            st.subheader("🔧 Select Features and Target")
            feature_cols = st.multiselect("✅ Feature columns:", numeric_cols, default=numeric_cols[:-1])
            target_col = st.selectbox("🎯 Target column:", options=numeric_cols, index=len(numeric_cols)-1)
            submitted = st.form_submit_button("🚀 Train Model & Predict")

        if submitted:
            if not feature_cols:
                st.error("Please select at least one feature column.")
            elif target_col in feature_cols:
                st.error("Target column must be different from features.")
            else:
                try:
                    # Clean Data
                    data = df[feature_cols + [target_col]].dropna()
                    X = data[feature_cols]
                    y = data[target_col]

                    # Train model
                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(X)
                    data['Predicted_' + target_col] = predictions

                    # Show metrics
                    r2 = r2_score(y, predictions)
                    mae = mean_absolute_error(y, predictions)
                    mse = mean_squared_error(y, predictions)
                    rmse = np.sqrt(mse)

                    st.sidebar.subheader("📊 Model Performance")
                    st.sidebar.metric("R² Score", f"{r2:.3f}")
                    st.sidebar.metric("MAE", f"{mae:.2f}")
                    st.sidebar.metric("RMSE", f"{rmse:.2f}")

                    # Display prediction table
                    st.subheader("📊 Prediction Results")
                    styled_table = data.style.highlight_max(axis=0, subset=['Predicted_' + target_col], color='lightgreen')
                    st.dataframe(styled_table, use_container_width=True)

                    # Plot
                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(x=y, y=predictions, ax=ax, color='blue', edgecolor='black')
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Line (y = x)')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    ax.legend()
                    st.pyplot(fig)

                    # Custom prediction
                    st.subheader("✍️ Custom Input for Prediction")
                    with st.form("custom_input_form"):
                        st.markdown("Enter values for each feature to predict the target:")
                        input_vals = {}
                        for col in feature_cols:
                            min_val = float(df[col].min())
                            max_val = float(df[col].max())
                            mean_val = float(df[col].mean())
                            input_vals[col] = st.number_input(f"{col}:", min_value=min_val, max_value=max_val, value=mean_val)
                        predict_submit = st.form_submit_button("🔮 Predict")

                    if predict_submit:
                        input_df = pd.DataFrame([input_vals])
                        result = model.predict(input_df)[0]
                        st.success(f"📢 Predicted `{target_col}`: **{result:.2f}**")

                    # Download CSV
                    st.download_button("📥 Download Results as CSV",
                                       data.to_csv(index=False),
                                       file_name="predictions.csv",
                                       mime="text/csv")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")
