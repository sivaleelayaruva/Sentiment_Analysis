import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def main():
    # Dataset
    data = {
        "text": [
            "I love this product", "Amazing quality", "Very bad experience",
            "I hate this", "It's okay", "Average item",
            "Excellent service", "Worst purchase", "Really happy",
            "Not worth it", "Fine product", "Superb", "Terrible",
            "Good but can improve", "Satisfied"
        ],
        "sentiment": [
            "positive", "positive", "negative", "negative", "neutral",
            "neutral", "positive", "negative", "positive", "negative",
            "neutral", "positive", "negative", "neutral", "positive"
        ]
    }

    df = pd.DataFrame(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, df["sentiment"])

    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("vectorizer.pkl", "wb") as vector_file:
        pickle.dump(vectorizer, vector_file)

    print("Model saved!")


if __name__ == "__main__":
    main()