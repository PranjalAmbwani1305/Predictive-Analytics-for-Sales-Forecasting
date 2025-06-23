import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting", layout="wide")

# Title
st.title("📊 Predictive Analytics for Sales Forecasting")

# Load data
@st.cache_data
def load_data():
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.warning("Please upload a dataset.")
        st.stop()

data = load_data()

# Show dataset preview
st.subheader("📄 Dataset Preview")
st.write(data.head())

# Check required columns
required_columns = ['Item_MRP', 'Outlet_Establishment_Year']
if not all(col in data.columns for col in required_columns):
    st.error(f"Dataset must include columns: {required_columns}")
    st.stop()

# Split features and target
X = data[['Item_MRP']]
y = data['Outlet_Establishment_Year']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
data['Predicted_Outlet_Establishment_Year'] = model.predict(X)

# R² score
r2 = r2_score(y, data['Predicted_Outlet_Establishment_Year'])

# Show prediction results
st.subheader("📋 Prediction Results")
st.write(data[['Item_MRP', 'Outlet_Establishment_Year', 'Predicted_Outlet_Establishment_Year']].head(10))

# Plot Actual vs Predicted
st.subheader("📈 Actual vs Predicted Sales")
fig, ax = plt.subplots()
ax.scatter(y, data['Predicted_Outlet_Establishment_Year'], edgecolor='black', alpha=0.75, label="Predicted Points")
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Line (y = x)')
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Actual vs Predicted Sales")
ax.legend()
st.pyplot(fig)

# Show R² Score
st.success(f"Model R² Score: {r2:.4f}")

# Section: Predict with custom input
st.subheader("🧮 Predict for Custom Input")
custom_mrp = st.number_input("Enter Item MRP:", min_value=0.0, format="%.2f")
if st.button("Predict Establishment Year"):
    custom_pred = model.predict(np.array([[custom_mrp]]))[0]
    st.info(f"Predicted Outlet Establishment Year: **{custom_pred:.2f}**")
