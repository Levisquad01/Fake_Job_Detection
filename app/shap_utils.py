import numpy as np
from scipy.sparse import hstack


def get_shap_explanation(model, tfidf, cleaned_text,
                         telecommuting,
                         has_logo,
                         has_questions):
    """
    Returns the top words contributing to the prediction.
    """

    text_vector = tfidf.transform([cleaned_text])

    structured = [[
        telecommuting,
        has_logo,
        has_questions
    ]]

    X = hstack([
        text_vector,
        structured
    ])

    feature_names = list(tfidf.get_feature_names_out())

    coef = model.coef_[0][:len(feature_names)]

    values = text_vector.toarray()[0]

    contributions = values * coef

    top_positive = np.argsort(contributions)[-10:][::-1]

    top_negative = np.argsort(contributions)[:10]

    positive_words = [
        feature_names[i]
        for i in top_positive
        if values[i] > 0
    ]

    negative_words = [
        feature_names[i]
        for i in top_negative
        if values[i] > 0
    ]

    return positive_words, negative_words