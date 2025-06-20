import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="🛍️ Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        st.error("❌ Your file does not contain numeric columns for training.")
    else:
        with st.form("select_columns"):
            st.subheader("🔧 Select Features and Target")
            col1, col2 = st.columns(2)
            feature_cols = col1.multiselect("✅ Select feature columns:", numeric_cols, help="Use only numeric columns")
            target_col = col2.selectbox("🎯 Select target column:", numeric_cols)

            submitted = st.form_submit_button("🚀 Run Prediction")

        if submitted:
            if not feature_cols:
                st.error("Please select at least one feature column.")
            elif target_col in feature_cols:
                st.error("Target column cannot be one of the features.")
            else:
                try:
                    # Prepare data
                    X = df[feature_cols]
                    y = df[target_col]
                    data = pd.concat([X, y], axis=1).dropna()
                    X = data[feature_cols]
                    y = data[target_col]

                    # Train model
                    model = LinearRegression()
                    model.fit(X, y)
                    data['Predicted'] = model.predict(X)

                    st.success("✅ Prediction completed!")

                    # Display prediction results
                    st.subheader("📊 Prediction Results")
                    data["Error"] = data["Predicted"] - data[target_col]
                    st.dataframe(data[feature_cols + [target_col, 'Predicted', 'Error']].round(2), use_container_width=True)

                    # Plot Actual vs Predicted
                    st.subheader("📈 Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(9, 6))
                    ax.scatter(data[target_col], data['Predicted'], color='skyblue', edgecolor='black', alpha=0.6, label='Predicted')
                    min_val = min(data[target_col].min(), data['Predicted'].min())
                    max_val = max(data[target_col].max(), data['Predicted'].max())
                    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Fit (y = x)')
                    ax.set_xlabel("Actual Sales")
                    ax.set_ylabel("Predicted Sales")
                    ax.set_title("Actual vs Predicted Sales using Linear Regression")
                    ax.legend()
                    st.pyplot(fig)

                    # Download CSV
                    st.download_button("📥 Download Predictions as CSV", data.to_csv(index=False), "predictions.csv", "text/csv")

                    # Custom input form
                    st.subheader("🔢 Predict with Custom Input")
                    with st.form("custom_input_form"):
                        input_vals = {}
                        cols = st.columns(len(feature_cols))
                        for i, col in enumerate(feature_cols):
                            default = float(df[col].mean())
                            input_vals[col] = cols[i].number_input(f"{col}", value=default)
                        custom_submit = st.form_submit_button("📡 Predict")

                        if custom_submit:
                            input_df = pd.DataFrame([input_vals])
                            custom_pred = model.predict(input_df)[0]
                            st.success(f"🎯 Predicted {target_col}: **{custom_pred:.2f}**")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
else:
    st.info("👆 Please upload a CSV file to begin.")
