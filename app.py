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
st.write(df.columns)

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

# -------------------------------
# FIND TARGET COLUMN AUTOMATICALLY
# -------------------------------
possible_targets = ["survival_status", "survival", "status", "outcome"]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    st.error("❌ No target column found (survival/status)")
    st.stop()


# -------------------------------
# MACHINE LEARNING MODEL (FINAL FIX)
# -------------------------------
st.subheader("🤖 Cancer Prediction Model")

if "survival_status" in df.columns:

    # ✅ DEFINE X AND y FIRST (THIS WAS MISSING)
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]

    # ✅ Use only numeric columns
    X = X.select_dtypes(include=["number"])

    # ✅ Handle missing values
    X = X.fillna(X.mean())

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # ✅ Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ✅ Train model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # ✅ Accuracy
    accuracy = model.score(X_test, y_test)

    st.success(f"✅ Model Accuracy: {round(accuracy*100,2)}%")

else:
    st.warning("❌ survival_status column not found")