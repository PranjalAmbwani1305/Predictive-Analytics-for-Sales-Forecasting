import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# App config
st.set_page_config(page_title="📈 Sales Forecast App", layout="wide")
st.title("📊 Sales Forecasting using Regression Models")

# File Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("The file must contain at least two numeric columns.")
        st.stop()

    # Sidebar config
    st.sidebar.subheader("⚙️ Configuration")
    feature_cols = st.sidebar.multiselect("Select Feature Columns", numeric_cols)
    target_col = st.sidebar.selectbox("Select Target Column", numeric_cols)

    model_type = st.sidebar.selectbox("🔧 Choose Model", ["Linear Regression", "Ridge", "Decision Tree"])
    split_data = st.sidebar.checkbox("🧪 Use Train/Test Split", value=True)
    test_size_pct = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)

    if feature_cols and target_col and target_col not in feature_cols:
        clean_df = df[feature_cols + [target_col]].dropna()
        X = clean_df[feature_cols]
        y = clean_df[target_col]

        if split_data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_pct/100, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Model selection
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Ridge":
            model = Ridge()
        else:
            model = DecisionTreeRegressor()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        st.sidebar.markdown("### 📈 Model Performance")
        st.sidebar.write(f"**R²:** {r2:.3f}")
        st.sidebar.write(f"**MAE:** {mae:.2f}")
        st.sidebar.write(f"**RMSE:** {rmse:.2f}")

        # Results table
        result_df = X_test.copy()
        result_df[target_col] = y_test
        result_df["Predicted"] = predictions
        result_df["Error"] = result_df[target_col] - result_df["Predicted"]

        st.subheader("🧾 Actual vs Predicted Table")
        styled_df = result_df.style.background_gradient(subset=["Error"], cmap="Reds")
        st.dataframe(styled_df)

        # Scatter Plot
        st.subheader("📊 Actual vs Predicted Scatter Plot")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(result_df[target_col], result_df["Predicted"], edgecolors='black', alpha=0.7)
        lims = [min(result_df[target_col].min(), result_df["Predicted"].min()),
                max(result_df[target_col].max(), result_df["Predicted"].max())]
        ax1.plot(lims, lims, 'r--', label='Ideal Line (y = x)')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title("Actual vs Predicted")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        # Residual Plot
        st.subheader("📉 Residual Plot")
        residuals = y_test - predictions
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(predictions, residuals, alpha=0.6)
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Analysis")
        ax2.grid(True)
        st.pyplot(fig2)

        # Feature Importance (only if model has coef_)
        if hasattr(model, 'coef_'):
            coef_df = pd.DataFrame({
                "Feature": feature_cols,
                "Coefficient": model.coef_,
                "Abs Coefficient": np.abs(model.coef_)
            }).sort_values("Abs Coefficient", ascending=False)
            st.subheader("📌 Feature Importance (Coefficients)")
            st.dataframe(coef_df[["Feature", "Coefficient"]])

        # Correlation Heatmap
        st.subheader("🔗 Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

        # Custom Prediction
        st.subheader("🎯 Predict for Custom Values")
        with st.form("custom_form"):
            custom_vals = {}
            for col in feature_cols:
                val = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))
                custom_vals[col] = val
            predict_btn = st.form_submit_button("📍 Predict Now")

        if predict_btn:
            input_df = pd.DataFrame([custom_vals])
            custom_result = model.predict(input_df)[0]
            st.metric(label=f"📍 Predicted {target_col}", value=f"{custom_result:.2f}")
            input_df[target_col] = custom_result
            st.download_button("📥 Download Custom Prediction", input_df.to_csv(index=False), "custom_prediction.csv")

        # Download predictions
        st.download_button("📁 Download Full Predictions", result_df.to_csv(index=False), "predictions.csv")

    else:
        st.warning("⚠️ Please select at least one feature and a target column (target ≠ feature).")
else:
    st.info("👈 Upload a CSV file to begin.")
