import numpy as np
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Evaluation")

st.markdown("""
This page summarizes the performance of the deployed Logistic Regression fake job detection model.
""")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "98.43%")
col2.metric("Precision", "80.95%")
col3.metric("Recall", "88.43%")
col4.metric("F1 Score", "84.53%")

cm = np.array([
    [3377, 25],
    [40, 306]
])

fig, ax = plt.subplots(figsize=(5,5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate","Fraud"]
)

disp.plot(ax=ax)

st.pyplot(fig)

st.subheader("ROC Curve")

st.image(
    "app/assets/roc_curve.png",
    use_container_width=True
)

st.subheader("Precision–Recall Curve")

st.image(
    "app/assets/pr_curve.png",
    use_container_width=True
)

report = pd.DataFrame({
    "Precision":[0.99,0.81],
    "Recall":[0.99,0.88],
    "F1":[0.99,0.85]
},
index=["Legitimate","Fraud"])

st.subheader("Classification Report")

st.dataframe(report, use_container_width=True)

st.subheader("Top Features")

top_features = pd.DataFrame({
    "Feature":[
        "urgent",
        "earn",
        "work home",
        "salary",
        "python",
        "experience"
    ],
    "Impact":[
        "Fraud",
        "Fraud",
        "Fraud",
        "Legitimate",
        "Legitimate",
        "Legitimate"
    ]
})

st.dataframe(top_features, use_container_width=True)

from PIL import Image

roc = Image.open("app/assets/roc_curve.png")
pr = Image.open("app/assets/pr_curve.png")

col1, col2 = st.columns(2)

with col1:
    st.image(roc, caption="ROC Curve")

with col2:
    st.image(pr, caption="Precision–Recall Curve")