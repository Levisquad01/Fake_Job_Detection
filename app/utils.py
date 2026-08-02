import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

def explain_prediction(model, vectorizer, text, top_n=10):

    feature_names = vectorizer.get_feature_names_out()

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()

    weights = model.coef_[0][:len(feature_names)]

    contributions = vector.multiply(weights)

    scores = contributions.toarray()[0]

    indices = scores.argsort()

    positive = indices[::-1][:top_n]

    negative = indices[:top_n]

    return (
        [(feature_names[i], scores[i]) for i in positive],
        [(feature_names[i], scores[i]) for i in negative]
    )

    print("Words:", len(feature_names))
    print("Weights:", len(weights))
    print("Vector:", vector.shape)