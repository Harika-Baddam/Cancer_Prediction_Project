import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Cancer Dashboard", layout="wide")
st.title("🧬 Colorectal Cancer Analysis & Prediction")


# -------------------------------
# LOAD DATA (AUTO DETECT FORMAT)
# -------------------------------
file_csv = "colorectal_cancer_prediction.csv"
file_excel = "colorectal_cancer_prediction.xlsx"

df = None

if os.path.exists(file_csv):
    df = pd.read_csv(file_csv)
elif os.path.exists(file_excel):
    try:
        df = pd.read_excel(file_excel, engine="openpyxl")
    except:
        st.error("❌ Excel file corrupted. Try saving again as CSV.")
        st.stop()
else:
    st.error("❌ Dataset not found in folder")
    st.stop()

# -------------------------------
# CLEAN COLUMN NAMES
# -------------------------------
df.columns = df.columns.str.lower().str.strip()

st.subheader("📂 Dataset Preview")
st.write(df.head())
st.write("Columns in dataset:", df.columns)

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

# -------------------------------
# AGE DISTRIBUTION
# -------------------------------
if "age" in filtered_df.columns:
    st.subheader("📊 Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["age"], kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------
# BMI DISTRIBUTION
# -------------------------------
if "bmi" in filtered_df.columns:
    st.subheader("⚖️ BMI Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["bmi"], kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------
# TIME TO RECURRENCE
# -------------------------------
if "time_to_recurrence" in filtered_df.columns:
    st.subheader("⏳ Time to Recurrence")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["time_to_recurrence"], kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------
# BOXPLOTS
# -------------------------------
if "survival_status" in filtered_df.columns and "age" in filtered_df.columns:
    st.subheader("📦 Age vs Survival Status")
    fig, ax = plt.subplots()
    sns.boxplot(x="survival_status", y="age", data=filtered_df, ax=ax)
    st.pyplot(fig)

if "recurrence" in filtered_df.columns and "bmi" in filtered_df.columns:
    st.subheader("📦 BMI vs Recurrence")
    fig, ax = plt.subplots()
    sns.boxplot(x="recurrence", y="bmi", data=filtered_df, ax=ax)
    st.pyplot(fig)

# -------------------------------
# COUNT PLOTS
# -------------------------------
if "chemotherapy_received" in filtered_df.columns and "survival_status" in filtered_df.columns:
    st.subheader("💊 Chemotherapy vs Survival")
    fig, ax = plt.subplots()
    sns.countplot(x="chemotherapy_received", hue="survival_status", data=filtered_df, ax=ax)
    st.pyplot(fig)

if "surgery_received" in filtered_df.columns and "recurrence" in filtered_df.columns:
    st.subheader("🏥 Surgery vs Recurrence")
    fig, ax = plt.subplots()
    sns.countplot(x="surgery_received", hue="recurrence", data=filtered_df, ax=ax)
    st.pyplot(fig)

# -------------------------------
# HEATMAP
# -------------------------------
st.subheader("🔥 Correlation Heatmap")

try:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(filtered_df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    st.pyplot(fig)
except:
    st.warning("Heatmap not available")

# -------------------------------
# MACHINE LEARNING MODEL
# -------------------------------
# -------------------------------
# FIND TARGET COLUMN (FIXED)
# -------------------------------
possible_targets = [
    "survival_status",
    "survival",
    "status",
    "outcome",
    "diagnosis",
    "result",
    "recurrence"
]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

# -------------------------------
# IF NOT FOUND → SHOW OPTIONS
# -------------------------------
if target_col is None:
    st.error("❌ No target column found")

    # 🔥 LET USER SELECT TARGET
    target_col = st.selectbox("Select Target Column", df.columns)

# -------------------------------
# MACHINE LEARNING
# -------------------------------
st.subheader("🤖 Cancer Prediction Model")

X = df.drop(columns=[target_col])
y = df[target_col]

# Convert categorical target if needed
if y.dtype == "object":
    y = y.astype("category").cat.codes

# Use numeric features only
X = X.select_dtypes(include=["number"])

# Fill missing values
X = X.fillna(X.mean())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.success(f"✅ Model Accuracy: {round(accuracy*100,2)}%")