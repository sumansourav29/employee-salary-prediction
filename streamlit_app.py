import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
try:
    df = pd.read_csv("adult.csv")
except FileNotFoundError:
    st.error("❌ 'adult.csv' not found. Make sure it's in the project directory.")
    st.stop()

# Get max values for numeric fields
max_vals = df.max(numeric_only=True).to_dict()

# Function to load or train the model
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

# Main app
def main():
    st.set_page_config(page_title="Employee Salary Predictor", layout="centered", initial_sidebar_state="expanded")

    with st.sidebar:
        st.title("💼 About This App")
        st.markdown("""
        - Predict whether an employee earns **>50K or <=50K** USD annually.
        - Based on demographic and work-related features.
        - Uses a **Random Forest Classifier** trained on the Adult dataset.
        - Developed by [Suman Sourav](https://github.com/sumansourav29)
        """)

    st.markdown("<h1 style='color:white; text-align:center;'>🧠 Employee Salary Predictor</h1>", unsafe_allow_html=True)

    model = load_or_train_model(df)

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

    # Input form
    st.subheader("🔍 Enter Details to Predict Income Bracket")
    inputs = {}

    for col in df.columns:
        if col != "income":
            if col in dropdown_options:
                default_option = "-- Select --"
                options = [default_option] + dropdown_options[col] + ["Other"]
                selection = st.selectbox(f"{col}", options)

                if selection == default_option:
                    inputs[col] = ""
                elif selection == "Other":
                    custom_value = st.text_input(f"Enter custom value for {col}")
                    inputs[col] = custom_value
                else:
                    inputs[col] = selection

            else:
                max_hint = f" (Max: {max_vals[col]})" if col in max_vals else ""
                value = st.text_input(f"{col}{max_hint}", placeholder=f"Enter {col}...")

                # Warning for exceeding max values
                try:
                    numeric_val = float(value)
                    if col in max_vals and numeric_val > max_vals[col]:
                        st.warning(f"⚠️ {col} exceeds expected maximum ({max_vals[col]}). Please verify.")
                except ValueError:
                    pass

                inputs[col] = value

    if st.button("🔮 Predict"):
        if "" in inputs.values():
            st.warning("⚠️ Please fill all fields before predicting.")
        else:
            try:
                input_df = pd.DataFrame([inputs])
                # Convert numeric columns
                for col in input_df.columns:
                    if col in max_vals:
                        input_df[col] = pd.to_numeric(input_df[col])

                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
                prediction = model.predict(input_df)[0]
                st.success(f"📈 Predicted Income Category: **{'More than 50K' if prediction == '>50K' else 'Less than or equal to 50K'} USD**")

                # Log the input and prediction
                log_data = pd.DataFrame([inputs])
                log_data["prediction"] = prediction
                if os.path.exists("prediction_log.csv"):
                    log_data.to_csv("prediction_log.csv", mode='a', header=False, index=False)
                else:
                    log_data.to_csv("prediction_log.csv", index=False)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# Run the app
if __name__ == "__main__":
    main()
