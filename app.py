import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data

try:
    df = pd.read_excel("colorectal_cancer_prediction.xlsx", engine="openpyxl")
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# -------------------------------
# CLEAN COLUMN NAMES
# -------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^\w]+", "_", regex=True)
)

# -------------------------------
# SHOW DATA
# -------------------------------
st.subheader("Dataset Preview")
st.write(df.head())
# -------------------------------
# SHOW DATA
# -------------------------------
st.subheader("Dataset Preview")
st.write(df.head())

# 🔥 IMPORTANT DEBUG
st.write("Columns:", list(df.columns))

# Check for missing values
df.isnull().sum()

#if missing
df.ffill(inplace=True)
# -------------------------------
# SIMPLE PLOT (SAFE)
# -------------------------------
if "age" in df.columns:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["age"], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.warning("No 'age' column found")


# Visualize survival status distribution
sns.countplot(x='survival_status', data=df)  # string
plt.title("Survival Status Distribution")
plt.show()

# Correlation heatmap (numerical features only)
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# EDA: Survival Status distribution
sns.countplot(x='survival_status', data=df) #string
plt.title("Survival Status Distribution")
plt.show()
# EDA: Correlation heatmap (numeric features only)
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

print(df.columns)

import seaborn as sns
import matplotlib.pyplot as plt
df_raw = df.copy()   # keep raw for EDA
df_model = pd.get_dummies(df, drop_first=True)  # encoded for ML

# -------------------------------
# Numeric feature distributions
# -------------------------------
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.histplot(df['bmi'], bins=30, kde=True)
plt.title("BMI Distribution")
plt.show()

sns.histplot(df['time_to_recurrence'], bins=30, kde=True)
plt.title("Time to Recurrence Distribution")
plt.show()

# -------------------------------
# Boxplots (numeric vs categorical)
# -------------------------------
sns.boxplot(x='survival_status', y='age', data=df) # string
plt.title("Age vs Survival Status")
plt.show()

sns.boxplot(x='recurrence', y='bmi', data=df)  # string
plt.title("BMI vs Recurrence")
plt.show()


# -------------------------------
# Treatment vs outcomes
# -------------------------------
sns.countplot(x='chemotherapy_received', hue='survival_status', data=df)  # string
plt.title("Chemotherapy vs Survival Status")
plt.show()

sns.countplot(x='Surgery_Received', hue='Recurrence', data=df)   # string
plt.title("Surgery vs Recurrence")
plt.show()

# -------------------------------
# Correlation heatmap (numeric only)
# -------------------------------
corr = df[['age','bmi','time_to_recurrence']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Preparing data for modeling
# Convert categorical
df = pd.get_dummies(df, drop_first=True)

target_col='Survival_Status' # or 'Recurrence' 
# Split
X_encoded = pd.get_dummies(X, drop_first=True) # encode features
X = df.drop(target_col, axis=1) #features
y = df[target_col]              #target

# Train Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score, classification_report   
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

