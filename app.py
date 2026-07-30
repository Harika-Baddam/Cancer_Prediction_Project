import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Title
st.title("🧬 Cancer Prediction App")
st.write("Predict whether a tumor is Benign or Malignant")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Age Distribution
st.subheader("📊 Age Distribution")

if "age" in X.columns:
    fig, ax = plt.subplots()
    sns.histplot(X["age"], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.info("Age column not available in dataset")

# Survival Status
st.subheader("🧬 Survival Status Distribution")

y_series = pd.Series(y)

fig, ax = plt.subplots()
y_series.value_counts().plot(kind='bar', ax=ax)
ax.set_xticklabels(["Malignant", "Benign"], rotation=0)
st.pyplot(fig)

# Heatmap
st.subheader("🔥 Feature Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pd.DataFrame(X).corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

# BMI
st.subheader("⚖️ BMI Analysis")

if "bmi" in X.columns:
    fig, ax = plt.subplots()
    sns.histplot(X["bmi"], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.warning("BMI column not available in dataset")

# Time to reccurance
st.subheader("⏳ Time to Recurrence")

if "time_to_recurrence" in X.columns:
    fig, ax = plt.subplots()
    sns.histplot(X["time_to_recurrence"], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Time to recurrence data not available")

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=5000)
model.fit(X_scaled, y)

st.subheader("Enter Patient Data")

# Create input fields (only few important features for simplicity)
radius = st.number_input("Mean Radius", value=14.0)
texture = st.number_input("Mean Texture", value=20.0)
perimeter = st.number_input("Mean Perimeter", value=90.0)
area = st.number_input("Mean Area", value=600.0)
smoothness = st.number_input("Mean Smoothness", value=0.1)

# Create input dataframe
input_data = pd.DataFrame({
    'mean radius': [radius],
    'mean texture': [texture],
    'mean perimeter': [perimeter],
    'mean area': [area],
    'mean smoothness': [smoothness]
})

# Fill missing columns with 0 (important step)
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns
input_data = input_data[X.columns]

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Prediction: Benign (No Cancer)")
    else:
        st.error("⚠️ Prediction: Malignant (Cancer)")
