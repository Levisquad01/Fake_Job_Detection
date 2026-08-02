from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Fake Job Analytics Dashboard")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "fake_job_postings.csv"

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Jobs", len(df))

with col2:
    st.metric("Legitimate Jobs", (df["fraudulent"] == 0).sum())

with col3:
    st.metric("Fraudulent Jobs", (df["fraudulent"] == 1).sum())

st.divider()

st.subheader("📄 Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

st.subheader("📊 Fraud Distribution")

counts = (
    df["fraudulent"]
    .value_counts()
    .rename(index={0: "Legitimate", 1: "Fraudulent"})
)

fig = px.bar(
    x=counts.index,
    y=counts.values,
    labels={"x": "Class", "y": "Count"},
    text=counts.values,
    title="Distribution of Job Postings"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.pie(
    values=counts.values,
    names=counts.index,
    title="Job Posting Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("📌 Missing Values")

missing = (
    df.isnull()
      .sum()
      .sort_values(ascending=False)
)

st.dataframe(
    missing.to_frame("Missing Values"),
    use_container_width=True
)

st.divider()

st.subheader("🌍 Top Hiring Locations")

top_locations = (
    df["location"]
      .fillna("Unknown")
      .value_counts()
      .head(10)
)

fig = px.bar(
    x=top_locations.values,
    y=top_locations.index,
    orientation="h",
    title="Top 10 Locations"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("💼 Employment Type")

employment = (
    df["employment_type"]
      .fillna("Unknown")
      .value_counts()
)

fig = px.pie(
    values=employment.values,
    names=employment.index
)

st.plotly_chart(fig, use_container_width=True)

import plotly.express as px

counts = df["fraudulent"].value_counts()

fig = px.pie(
    values=counts.values,
    names=["Legitimate", "Fraud"],
    title="Job Distribution"
)

st.plotly_chart(fig, use_container_width=True)

industry = df["employment_type"].value_counts().head(10)

fig = px.bar(
    x=industry.index,
    y=industry.values,
    title="Employment Types"
)

st.plotly_chart(fig, use_container_width=True)

