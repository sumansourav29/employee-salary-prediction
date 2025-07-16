import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_or_train_model(data):
    try:
        model = joblib.load("best_model.pkl")
        return model
    except Exception as e:
        st.warning(f"⚠️ Couldn't load model. Training new one: {e}")
        X = pd.get_dummies(data.drop("Salary", axis=1))
        y = data["Salary"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, "best_model.pkl")
        return model

def main():
    st.title("💼 Employee Salary Predictor")

    try:
        df = pd.read_csv("adult.csv")
    except FileNotFoundError:
        st.error("❌ 'adult.csv' not found.")
        return

    model = load_or_train_model(df)

    st.subheader("🔍 Predict Salary")
    inputs = {}
    for col in df.columns:
        if col != "Salary":
            inputs[col] = st.text_input(col)

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([inputs])
            input_df = pd.get_dummies(input_df)
            # Match columns to training data
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
            prediction = model.predict(input_df)
            st.success(f"💰 Predicted Salary: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
