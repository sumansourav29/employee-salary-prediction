import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# Background styling (optional - CSS for color or image)
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f2f2f2;
            background-image: linear-gradient(to bottom right, #dbeafe, #fef3c7);
            font-family: 'Segoe UI', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True
    )
set_bg()

# Sidebar info
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    This app predicts whether a person's income exceeds **$50K/year** based on their attributes.

    🧠 Built with:
    - Streamlit
    - Scikit-learn
    - Random Forest Classifier
    """)

def load_or_train_model(data):
    if "income" not in data.columns:
        st.error("❌ 'income' column not found. Please check the dataset format.")
        st.stop()

    try:
        model = joblib.load("best_model.pkl")
        return model
    except Exception:
        st.warning("⚠️ Couldn't load model. Training a new one...")
        X = pd.get_dummies(data.drop("income", axis=1))
        y = data["income"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, "best_model.pkl")
        return model

def main():
    st.title("🧠 Employee Salary Prediction")
    st.markdown("### 👇 Fill in the details below to predict income bracket:")

    try:
        df = pd.read_csv("adult.csv")
    except FileNotFoundError:
        st.error("❌ 'adult.csv' not found. Make sure it's in the repo.")
        return

    model = load_or_train_model(df)

    with st.expander("📂 View sample data"):
        st.dataframe(df.head())

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
        "native-country": ['United-States', 'Peru', 'Guatemala', 'Mexico', 'Dominican-Republic', 'Ireland',
                           'Germany', 'Philippines', 'Thailand', 'Haiti', 'El-Salvador', 'Puerto-Rico',
                           'Vietnam', 'South', 'Columbia', 'Japan', 'India', 'Cambodia', 'Poland', 'Laos',
                           'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal', 'China', 'Nicaragua',
                           'Honduras', 'Iran', 'Scotland', 'Jamaica', 'Ecuador', 'Yugoslavia', 'Hungary',
                           'Hong', 'Greece', 'Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'France',
                           'Holand-Netherlands']
    }

    inputs = {}
    for col in df.columns:
        if col != "income":
            if col in dropdown_options:
                options = dropdown_options[col] + ["Others"]
                selection = st.selectbox(f"🔽 {col}", options)

                if selection == "Others":
                    custom_value = st.text_input(f"✍️ Enter custom value for {col}")
                    inputs[col] = custom_value
                else:
                    inputs[col] = selection
            else:
                inputs[col] = st.text_input(f"✍️ {col}")

    if st.button("🚀 Predict"):
        try:
            input_df = pd.DataFrame([inputs])
            input_df_encoded = pd.get_dummies(input_df)
            input_df_encoded = input_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
            prediction = model.predict(input_df_encoded)
            st.toast(f"🎯 Predicted Income Category: {prediction[0]}", icon="💰")

            # Optional: Save user input
            input_df["income"] = "unknown"
            df = pd.concat([df, input_df], ignore_index=True)
            df.to_csv("adult.csv", index=False)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
