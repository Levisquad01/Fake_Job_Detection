from pathlib import Path
import streamlit as st
import joblib
from scipy.sparse import hstack

from utils import clean_text
from shap_utils import get_shap_explanation

st.set_page_config(
    page_title="Fake Job Posting Detection",
    page_icon="💼",
    layout="wide"
)

st.sidebar.title("💼 Fake Job Detection")

st.sidebar.success("✅ Model Loaded Successfully")

st.sidebar.markdown("""
### 🤖 About

This application predicts whether a job posting is:

- ✅ Legitimate
- 🚨 Fraudulent

using a **Logistic Regression** model trained on thousands of real and fraudulent job postings.

The model uses:

- TF-IDF text features
- Job metadata
- Natural Language Processing (NLP)

---

### 📊 Deployed Model

- Logistic Regression
- Accuracy: **98.43%**
- Precision: **80.95%**
- Recall: **88.43%**
- F1 Score: **84.53%**

**Why this model?**

Provides prediction probabilities, making confidence scores possible while maintaining strong fraud detection.

---

### 👨‍💻 Developer

Gopal Kumar Rajwar

AI/ML Internship Project
""")

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Load model
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)
feature_names = tfidf.get_feature_names_out()
#st.write(type(model))
#st.write(model)



st.title("💼 Fake Job Posting Detection System")

st.markdown("""
### AI-Powered Recruitment Fraud Detection

This application analyzes job advertisements using **Natural Language Processing (NLP)** and **Machine Learning** to identify potentially fraudulent job postings.

Enter the job information below and click **Predict**.
""")

st.divider()



col1, col2 = st.columns(2)

title = st.text_input("Job Title")

company = st.text_area("Company Profile")

description = st.text_area("Job Description")

requirements = st.text_area("Requirements")

benefits = st.text_area("Benefits")

col1, col2, col3 = st.columns(3)

with col1:
    telecommuting = st.selectbox(
        "Remote",
        [0,1]
    )

with col2:
    has_logo = st.selectbox(
        "Company Logo",
        [0,1]
    )

with col3:
    has_questions = st.selectbox(
        "Has Screening Questions",
        [0,1]
    )

button_col1, button_col2 = st.columns(2)

with button_col1:
    predict = st.button("🔍 Predict")

with button_col2:
    clear = st.button("🗑️ Clear Form")

if clear:
    st.rerun()

if predict:

    with st.spinner("Analyzing job posting..."):

        text = " ".join([
            title,
            company,
            description,
            requirements,
            benefits
        ])

        cleaned = clean_text(text)

        text_vector = tfidf.transform([cleaned])

        structured = [[
            telecommuting,
            has_logo,
            has_questions
        ]]

        X = hstack([
            text_vector,
            structured
        ])

    if len(description.strip()) < 30:
        st.warning(
        "Please enter a more detailed job description."
    )
        st.stop()
    prediction = model.predict(X)[0]

    

    positive_words, negative_words = get_shap_explanation(
    model,
    tfidf,
    cleaned,
    telecommuting,
    has_logo,
    has_questions
)

    probabilities = model.predict_proba(X)[0]

    #st.write("Prediction probabilities:", probabilities)

    confidence = probabilities[prediction] * 100

    if prediction == 0:
        risk = "🟢 Low Risk"
    elif confidence >= 90:
        risk = "🔴 High Risk"
    elif confidence >= 70:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟡 Low-Medium Risk"

    st.metric(
    label="Risk Level",
    value=risk
)    

    if prediction == 0:

        st.success("✅ Legitimate Job Posting")

        st.info("""
This posting appears to be genuine.

Always verify company details before applying.
""")

        st.subheader("✅ Why this looks legitimate")

        for word in positive_words:
            st.write("•", word)

    else:

        st.error("🚨 Fraudulent Job Posting")

        st.warning("""
This posting contains characteristics commonly associated with fraudulent job advertisements.
""")

        st.subheader("🚨 Why this looks fraudulent")

        for word in negative_words:
            st.write("•", word)

    st.subheader("Prediction Confidence")

    st.progress(confidence / 100)

    st.metric(
    "Confidence",
    f"{confidence:.2f}%"
)

st.divider()

st.divider()

st.caption("""
Built with ❤️ using

Python • Scikit-learn • Streamlit • NLP • TF-IDF • Logistic Regression

AI/ML Internship Project

Developed by **Gopal Kumar Rajwar**
""")


