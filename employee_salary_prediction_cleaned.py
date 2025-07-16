#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv(r"C:\Users\Nanthini M\Downloads\Edunet\AI\Implementation\adult.csv")


# In[3]:


data.head(10)


# In[4]:


data.tail(3)


# In[5]:


data.shape


# In[6]:


#null values
data.isna().sum() #mean mdeian mode arbitrary


# In[7]:


print(data.workclass.value_counts())


# In[8]:


data.workclass.replace({'?':'Others'},inplace=True)
print(data['workclass'].value_counts())


# In[9]:


print(data['occupation'].value_counts())


# In[10]:


data.occupation.replace({'?':'Others'},inplace=True)
print(data['occupation'].value_counts())


# In[11]:


data=data[data['workclass']!='Without-pay']
data=data[data['workclass']!='Never-worked']
print(data['workclass'].value_counts())


# In[12]:


print(data.relationship.value_counts())


# In[13]:


print(data.gender.value_counts())


# In[14]:


data.shape


# In[15]:


#outlier detection
import matplotlib.pyplot as plt   #visualization
plt.boxplot(data['age'])
plt.show()


# In[16]:


data=data[(data['age']<=75)&(data['age']>=17)]


# In[17]:


plt.boxplot(data['age'])
plt.show()


# In[18]:


data.shape


# In[19]:


plt.boxplot(data['capital-gain'])
plt.show()


# In[20]:


plt.boxplot(data['capital-gain'])
plt.show()


# In[21]:


plt.boxplot(data['educational-num'])
plt.show()


# In[22]:


data=data[(data['educational-num']<=16)&(data['educational-num']>=5)]


# In[23]:


plt.boxplot(data['educational-num'])
plt.show()


# In[24]:


plt.boxplot(data['hours-per-week'])
plt.show()


# In[25]:


data.shape


# In[26]:


data=data.drop(columns=['education']) #redundant features removal


# In[27]:


data


# In[28]:


from sklearn.preprocessing import LabelEncoder   #import libarary
encoder=LabelEncoder()                       #create object
data['workclass']=encoder.fit_transform(data['workclass']) #7 categories   0,1, 2, 3, 4, 5, 6,
data['marital-status']=encoder.fit_transform(data['marital-status'])   #3 categories 0, 1, 2
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])      #5 categories  0, 1, 2, 3, 4
data['race']=encoder.fit_transform(data['race'])  
data['gender']=encoder.fit_transform(data['gender'])    #2 catogories     0, 1
data['native-country']=encoder.fit_transform(data['native-country'])


# In[29]:


data


# In[30]:


x=data.drop(columns=['income'])
y=data['income']
x


# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


# In[32]:


import matplotlib.pyplot as plt
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("✅ Saved best model as best_model.pkl")


# In[42]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nimport joblib\n\n# Load the trained model\nmodel = joblib.load("best_model.pkl")\n\nst.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")\n\nst.title("💼 Employee Salary Classification App")\nst.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")\n\n# Sidebar inputs (these must match your training feature columns)\nst.sidebar.header("Input Employee Details")\n\n# ✨ Replace these fields with your dataset\'s actual input columns\nage = st.sidebar.slider("Age", 18, 65, 30)\neducation = st.sidebar.selectbox("Education Level", [\n    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"\n])\noccupation = st.sidebar.selectbox("Job Role", [\n    "Tech-support", "Craft-repair", "Other-service", "Sales",\n    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",\n    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",\n    "Protective-serv", "Armed-Forces"\n])\nhours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)\nexperience = st.sidebar.slider("Years of Experience", 0, 40, 5)\n\n# Build input DataFrame (⚠️ must match preprocessing of your training data)\ninput_df = pd.DataFrame({\n    \'age\': [age],\n    \'education\': [education],\n    \'occupation\': [occupation],\n    \'hours-per-week\': [hours_per_week],\n    \'experience\': [experience]\n})\n\nst.write("### 🔎 Input Data")\nst.write(input_df)\n\n# Predict button\nif st.button("Predict Salary Class"):\n    prediction = model.predict(input_df)\n    st.success(f"✅ Prediction: {prediction[0]}")\n\n# Batch prediction\nst.markdown("---")\nst.markdown("#### 📂 Batch Prediction")\nuploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")\n\nif uploaded_file is not None:\n    batch_data = pd.read_csv(uploaded_file)\n    st.write("Uploaded data preview:", batch_data.head())\n    batch_preds = model.predict(batch_data)\n    batch_data[\'PredictedClass\'] = batch_preds\n    st.write("✅ Predictions:")\n    st.write(batch_data.head())\n    csv = batch_data.to_csv(index=False).encode(\'utf-8\')\n    st.download_button("Download Predictions CSV", csv, file_name=\'predicted_classes.csv\', mime=\'text/csv\')\n\n')


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[ ]:




