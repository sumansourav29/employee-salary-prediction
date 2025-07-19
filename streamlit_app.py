import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page config and dark theme
st.set_page_config(
    page_title="Employee Salary Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Dark theme custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .css-18e3th9, .css-1d391kg {
        background-color: #0e1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load or train model
def load_or_train_model(data):
    if "income" not in data.columns:
        st.error("❌ 'income' column not found in dataset.")
        st.stop()
    try:
        model = joblib.load("best_model.pkl")
        return model
    except:
        st.warning("⚠️ Couldn't load model. Training a new one...")
        X = pd.get_dummies(data.drop("income", axis=1))
        y = data["income"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, "best_model.pkl")
        return model

# Sidebar Information
st.sidebar.title("ℹ️ About the App")
st.sidebar.markdown("""
This app predicts whether a person's annual income is **more than $50K** or **less than or equal to $50K** using a machine learning model trained on the **UCI Adult Income Dataset**.

👤 Built by: [Suman Sourav Sahoo](https://github.com/sumansourav29)

📂 Model: Random Forest Classifier  
📊 Dataset: `adult.csv`  
🧠 Features: workclass, education, marital-status, occupation, etc.
""")

# Main function
def main():
    st.title("🧠 Employee Salary Predictor")

    # Load dataset
    try:
        df = pd.read_csv("adult.csv")
    except FileNotFoundError:
        st.error("❌ 'adult.csv' not found.")
        return

    model = load_or_train_model(df)

    st.subheader("🔍 Enter Applicant Details")

    dropdown_options = {
        "workclass": ['Private', 'Local-gov', 'Self-emp-not-inc', 'Federal-gov',
                      'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
        "education": ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', 'Prof-school',
                      'Bachelors', 'Masters', 'Doctorate', 'Assoc-voc', '12th', 'Preschool'],
        "marital-status": ['Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced',
                           'Separated', 'Married-spouse-absent', 'Married-AF-spouse'],
        "occupation": ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Other-service',
                       'Prof-specialty', 'Craft-repair', 'Adm-clerical', 'Exec-managerial',
                       'Tech-support', 'Sales', 'Priv-house-serv', 'Transport-moving',
                       'Handlers-cleaners', 'Armed-Forces'],
        "relationship": ['Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative'],
        "race": ['Black', 'White', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo'],
        "gender": ['Male', 'Female'],
        "native-country": ['United-States', 'India', 'Philippines', 'Germany', 'Mexico', 'Canada']
    }

    inputs = {}
    for col in df.columns:
        if col != "income":
            if col in dropdown_options:
                options = dropdown_options[col] + ["Other"]
                selection = st.selectbox(f"{col}", options)
                if selection == "Other":
                    custom_val = st.text_input(f"Enter custom value for {col}")
                    inputs[col] = custom_val
                else:
                    inputs[col] = selection
            else:
                inputs[col] = st.text_input(f"{col}", placeholder=f"Enter {col}...")

    if st.button("🚀 Predict"):
        try:
            input_df = pd.DataFrame([inputs])
            input_df_encoded = pd.get_dummies(input_df)
            input_df_encoded = input_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
            prediction = model.predict(input_df_encoded)

            # Friendly message
            result_text = (
                "More than $50K" if prediction[0].strip().startswith(">") else
                "Less than or equal to $50K"
            )
            st.toast(f"🎯 Predicted Income: **{result_text}**", icon="💰")

            # Log prediction
            log_data = input_df.copy()
            log_data["prediction"] = result_text
            if not os.path.exists("prediction_logs.csv"):
                log_data.to_csv("prediction_logs.csv", index=False)
            else:
                log_data.to_csv("prediction_logs.csv", mode='a', header=False, index=False)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
