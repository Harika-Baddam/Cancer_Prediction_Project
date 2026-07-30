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
df = pd.read_csv("colorectal_cancer_prediction.csv")
df.columns = df.columns.str.lower().str.strip()
st.write(df.head())
X = df.drop(columns=["survival_status"])
y = df["survival_status"]

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
st.subheader("⚖️ BMI Distribution")

if 'bmi' in df.columns:
    st.write("BMI available")
    st.write("Columns:", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df['bmi'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.error("BMI column not found")

# Time to reccurance
st.subheader("⏳ Time_to_Recurrence")

if 'time_to_recurrence' in df.columns:
    fig, ax = plt.subplots()
    sns.histplot(df['time_to_recurrence'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.error("Time_to_recurrence column not found")

# Treatment vs Outcomes
# -------------------------------
st.subheader("💊 Chemotherapy vs Survival")

fig, ax = plt.subplots()
sns.countplot(x='Chemotherapy_Received', hue='Survival_Status', data=df, ax=ax)
ax.set_title("Chemotherapy vs Survival Status")
st.pyplot(fig)

st.subheader("🏥 Surgery vs Recurrence")

fig, ax = plt.subplots()
sns.countplot(x='Surgery_Received', hue='Recurrence', data=df, ax=ax)
ax.set_title("Surgery vs Recurrence")
st.pyplot(fig)

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
