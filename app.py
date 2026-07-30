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

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
st.sidebar.header("🔍 Filter Data")

if "survival_status" in df.columns:
    selected_status = st.sidebar.selectbox(
        "Select Survival Status",
        df["survival_status"].unique()
    )
    filtered_df = df[df["survival_status"] == selected_status]
else:
    filtered_df = df

# Age Distribution
st.subheader("📊 Age Distribution")

if "age" in X.columns:
    fig, ax = plt.subplots()
    sns.histplot(X["age"], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.info("Age column not available in dataset")


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

# -------------------------------
# BOXPLOTS
# -------------------------------

st.subheader("📦 Age vs Survival Status")

if "survival_status" in filtered_df.columns and "age" in filtered_df.columns:
    fig, ax = plt.subplots()
    sns.boxplot(x="survival_status", y="age", data=filtered_df, ax=ax)
    st.pyplot(fig)

st.subheader("📦 BMI vs Recurrence")

if "recurrence" in filtered_df.columns and "bmi" in filtered_df.columns:
    fig, ax = plt.subplots()
    sns.boxplot(x="recurrence", y="bmi", data=filtered_df, ax=ax)
    st.pyplot(fig)

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
