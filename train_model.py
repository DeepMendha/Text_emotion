import re
import nltk
import pandas as pd
import pickle

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def preprocess(text, regex, stopwords):
    text = text.lower()
    text = re.sub(regex, " ", text)
    tokens = [each for each in word_tokenize(text) if each not in stopwords]
    return " ".join(tokens)


def export_model(model, file_name: str) -> None:
    s = pickle.dumps(model)
    with open(file_name, "wb") as f:
        f.write(s)

    return f"Model written to file {file_name}"


def get_x_y(filename, regex, stopwords):
    # create DF from CSV file.
    df = pd.read_csv(filename)
    # process the data
    df["content"] = df["content"].apply(
        lambda row: preprocess(row, regex, stopwords)
    )

    # Split data into X and Y
    X = df["content"]
    Y = df["sentiment"]
    return X, Y


if __name__ == "__main__":

    # download required NLTK packages
    nltk.download("stopwords")
    nltk.download("punkt")

    # get stopwords from nltk
    stopwords = nltk.corpus.stopwords.words("english")

    # regex to remove non-Alphanumeric and whitespaces values
    regex = re.compile(r"[^A-Za-z0-9 ]")

    data_file = "./text_emo.csv"
    X, Y = get_x_y(data_file, regex, stopwords)

    # Split X and Y into Training and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=123
    )

    # TFIDF Vector
    tfidf_vector = TfidfVectorizer(ngram_range=(1, 3))

    # Multi-layer Percepton
    mlp_classifier = MLPClassifier(
        early_stopping=True,
        learning_rate="invscaling"
    )

    # Create Pipeline
    stages = [("tfidf", tfidf_vector), ("classifier", mlp_classifier)]
    pipeline_model = Pipeline(stages)

    # train pipeline model
    pipeline_model.fit(x_train, y_train)

    # Get prediction
    y_pred = pipeline_model.predict(x_test)

    # get classification report
    print(classification_report(y_test, y_pred))

    # export MLP Model
    export_model(pipeline_model, "./mlp_classifier.pk")
