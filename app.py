import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# App configuration
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction App")

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    st.write("🧾 Available columns:")
    st.write(df.columns.tolist())

    # Numeric columns only
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        st.error("❌ Your file does not contain numeric columns for training.")
    else:
        with st.form("select_columns"):
            st.subheader("🔧 Select Features and Target")

            # Feature multiselect
            feature_cols = st.multiselect("✅ Select feature columns (numeric only):", numeric_cols)

            # Styling the target column selectbox
            st.markdown(
                """
                <style>
                div[data-testid="stSelectbox"] div[data-baseweb="select"] {
                    border: 2px solid red !important;
                    border-radius: 5px;
                    padding: 5px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Set default target column to "Item_Outlet_Sales" if exists
            default_target = "Item_Outlet_Sales" if "Item_Outlet_Sales" in numeric_cols else numeric_cols[0]
            target_col = st.selectbox("🎯 Select target column:", numeric_cols, index=numeric_cols.index(default_target))

            submitted = st.form_submit_button("🚀 Run Prediction")

        if submitted:
            if not feature_cols:
                st.error("Please select at least one feature column.")
            elif target_col in feature_cols:
                st.error("Target column cannot be one of the features.")
            else:
                try:
                    # Data prep
                    X = df[feature_cols]
                    y = df[target_col]
                    data = pd.concat([X, y], axis=1).dropna()
                    X = data[feature_cols]
                    y = data[target_col]

                    # Train model
                    model = LinearRegression()
                    model.fit(X, y)

                    # Predict
                    data['Predicted'] = model.predict(X)

                    st.success("✅ Prediction completed!")
                    st.subheader("📊 Prediction Results")
                    st.write(data[feature_cols + [target_col, 'Predicted']])

                    # Visualization
                    st.subheader("📈 Actual vs Predicted Sales")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.set(style="whitegrid")

                    sns.scatterplot(
                        x=data[target_col],
                        y=data['Predicted'],
                        ax=ax,
                        color='royalblue',
                        edgecolor='black',
                        alpha=0.6,
                        s=70
                    )

                    min_val = min(data[target_col].min(), data['Predicted'].min())
                    max_val = max(data[target_col].max(), data['Predicted'].max())
                    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Fit (y = x)')

                    ax.set_title("Actual vs Predicted Sales using Linear Regression", fontsize=16, weight='bold')
                    ax.set_xlabel("Actual Sales", fontsize=12)
                    ax.set_ylabel("Predicted Sales", fontsize=12)
                    ax.legend(title="Legend", loc="upper left", fontsize=10)
                    ax.set_xlim(min_val, max_val)
                    ax.set_ylim(min_val, max_val)

                    st.pyplot(fig)

                    # Download option
                    st.download_button(
                        "📥 Download Predictions as CSV",
                        data.to_csv(index=False),
                        "predictions.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
else:
    st.info("📎 Please upload a CSV file to begin.")
