import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")

st.title("Customer Purchase Prediction System")
st.write("This application predicts whether a customer will purchase a product using a balanced machine learning model.")

data_url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/refs/heads/main/Customer%20Purchase.csv"

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

df = load_data(data_url)

st.subheader(" Dataset Preview")
st.dataframe(df.head())

df.columns = df.columns.str.strip()
df["Purchased"] = df["Purchased"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Customer ID", "Purchased"])
y = df["Purchased"]

X = pd.get_dummies(X, columns=["Gender", "Education", "Review"], drop_first=True)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader(" Model Performance")
st.write(f"**Accuracy:** {round(accuracy * 100, 2)} %")

st.subheader(" Classification Report")
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report_df)

st.subheader(" Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.dataframe(pd.DataFrame(cm, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"]))

st.subheader("🧪 Predict for a New Customer")

age = st.number_input("Age", 18, 70, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education", ["School", "UG", "PG"])
review = st.selectbox("Review", ["Poor", "Average", "Good"])

input_data = pd.DataFrame({
    "Age": [age],
    "Gender_" + gender: [1],
    "Education_" + education: [1],
    "Review_" + review: [1]
})

for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]
input_scaled = scaler.transform(input_data)

if st.button("Predict Purchase"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("🛒 Prediction: Customer WILL Purchase")
    else:
        st.error("❌ Prediction: Customer will NOT Purchase")
