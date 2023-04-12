import re
import pickle
import nltk
import streamlit as st

from nltk import word_tokenize


def preprocess(text, regex, stopwords):
    text = text.lower()
    text = re.sub(regex, " ", text)
    tokens = [each for each in word_tokenize(text) if each not in stopwords]
    return " ".join(tokens)


def read_model(file_name):
    with open(file_name, "rb") as f:
        mdl = pickle.loads(f.read())
    return mdl


if __name__ == "__main__":

    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')
        nltk.download('stopwords')

    st.write("# Text Emotions Prediction")
    t1 = st.text_input("Enter any text: ")

    regex = re.compile(r"[^A-Za-z0-9 ]")
    nltk.download("stopwords")
    stopwords = nltk.corpus.stopwords.words("english")

    pipeline_model_file = "./mlp_classifier.pk"

    # Get pipeline model
    pipeline_model = read_model(pipeline_model_file)
    
    emoji_dict = {
        "joy":"😂", "fear":"😱", "anger":"😠",
        "sadness":"😢", "disgust":"😒", "shame":"😳", 
        "guilt":"😳", "neutral":"😐", "happiness": "🙂", 
        "worry": "😟", "love": "😍", "surprise": "😲",
        "fun": "😄", "relief": "😌", "hate": "😡",
        "empty": "😑", "enthusiasm": "🙌", "boredom": "🥱"
    }

    texts = [t1]
    
    for text in texts:
        preprocess_text = preprocess(text, regex, stopwords)
        pred_emoji = pipeline_model.predict([preprocess_text])[0]
        # pred_emoji = mlp_model.predict(x)
        st.write(pred_emoji, emoji_dict[pred_emoji.lower()])
