import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

data = load_data()

# Define target and features
target_col = "Item_Outlet_Sales"  # <- change this based on your dataset
features = ["Item_MRP", "Outlet_Establishment_Year"]  # Add more features as needed

st.title("📈 Predictive Analytics App - Sales Forecast")

# Show raw data
with st.expander("🔍 Show Raw Data"):
    st.write(data.head())

# Handle missing values
if data[features + [target_col]].isnull().any().any():
    data = data.dropna(subset=features + [target_col])

# Prepare data
X = data[features]
y = data[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, predictions)
st.subheader("Model Performance")
st.write(f"R-squared (R2) Score: `{r2:.4f}`")

# Plot: Actual vs Predicted
def plot_actual_vs_predicted(actual, predicted, target_label):
    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, edgecolor='k', alpha=0.7, label="Predicted Points")
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Ideal Line (y = x)')
    ax.set_xlabel(f"Actual {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(f"Actual vs Predicted {target_label}")
    ax.legend()
    st.pyplot(fig)

plot_actual_vs_predicted(y_test, predictions, target_col)

# Custom Prediction Input
st.subheader("🔮 Predict Custom Input")
input_data = {}

for col in features:
    val = st.number_input(f"Enter {col}", value=float(data[col].mean()))
    input_data[col] = val

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    custom_pred = model.predict(input_df)[0]
    st.success(f"Predicted {target_col}: `{custom_pred:.2f}`")

# Prediction Table (Optional)
st.subheader("📊 Sample Predictions")
sample = X_test.copy()
sample[target_col] = y_test
sample["Predicted"] = predictions
st.dataframe(sample.head(10))
