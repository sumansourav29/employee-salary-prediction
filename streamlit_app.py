import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("adult.csv")

df = load_data()

# Max values for numeric inputs
max_vals = {
    "fnlwgt": int(df["fnlwgt"].max()),
    "education-num": int(df["education-num"].max()),
    "hours-per-week": int(df["hours-per-week"].max())
}

# Dropdown options
dropdown_options = {
    "workclass": df["workclass"].dropna().unique().tolist(),
    "education": df["education"].dropna().unique().tolist(),
    "marital-status": df["marital-status"].dropna().unique().tolist(),
    "occupation": df["occupation"].dropna().unique().tolist(),
    "relationship": df["relationship"].dropna().unique().tolist(),
    "race": df["race"].dropna().unique().tolist(),
    "gender": df["gender"].dropna().unique().tolist(),
    "native-country": df["native-country"].dropna().unique().tolist()
}

# Train or load model
def load_or_train_model(data):
    if "income" not in data.columns:
        st.error("❌ 'income' column not found. Please check the dataset format.")
        st.stop()

    try:
        model = joblib.load("best_model.pkl")
    except:
        st.warning("Training a new model...")
        X = pd.get_dummies(data.drop("income", axis=1))
        y = data["income"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, "best_model.pkl")
    return model

model = load_or_train_model(df)

# Streamlit UI
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")
st.title("💼 Employee Salary Predictor")

st.sidebar.title("📘 About")
st.sidebar.info("""
This app predicts whether an individual's income is more or less than $50K/year based on demographic data.

Developed by: [Your Name]
Data: UCI Adult Income Dataset
""")

st.subheader("🔍 Fill the details below")

inputs = {}
for col in df.columns:
    if col != "income":
        if col in dropdown_options:
            options = [""] + dropdown_options[col] + ["Others"]
            selection = st.selectbox(f"🔽 {col}", options, index=0, key=col)

            if selection == "Others":
                custom_value = st.text_input(f"✍️ Enter custom value for {col}", key=f"{col}_custom")
                inputs[col] = custom_value
            elif selection != "":
                inputs[col] = selection
        else:
            if col in max_vals:
                value = st.text_input(f"✍️ {col} (Max: {max_vals[col]})", key=col)
                if value:
                    try:
                        val = int(value)
                        if val > max_vals[col]:
                            st.warning(f"⚠️ {col} cannot exceed {max_vals[col]}")
                        inputs[col] = val
                    except ValueError:
                        st.error(f"❌ {col} must be a number.")
            else:
                inputs[col] = st.text_input(f"✍️ {col}", key=col)

if st.button("🔮 Predict"):
    try:
        input_df = pd.DataFrame([inputs])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(input_df)
        st.success(f"📈 Predicted Income: {'>50K' if prediction[0] == '>50K' else '<=50K'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
