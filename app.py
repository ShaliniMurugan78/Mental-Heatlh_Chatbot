# ===================== IMPORTS =====================

import nltk
import json
import random
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

from transformers import pipeline

import spacy
import spacy.cli
from spacy.language import Language
from spacy_langdetect import LanguageDetector

from flask import Flask, render_template, request


# ==================================================
# NLTK SETUP
# ==================================================

lemmatizer = WordNetLemmatizer()


# ==================================================
# LOAD MODELS (LOAD ONLY ONCE → FAST + PRODUCTION SAFE)
# ==================================================

print("Loading ML model...")
model = load_model("model.h5")

print("Loading intents and labels...")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("texts.pkl", "rb"))
classes = pickle.load(open("labels.pkl", "rb"))


# ==================================================
# TRANSLATION MODELS (lighter pipeline loading)
# ==================================================

print("Loading translation models...")

eng_swa_translator = pipeline(
    "text2text-generation",
    model="Rogendo/en-sw"
)

swa_eng_translator = pipeline(
    "text2text-generation",
    model="Rogendo/sw-en"
)


def translate_text_eng_swa(text):
    return eng_swa_translator(text, max_length=128)[0]["generated_text"]


def translate_text_swa_eng(text):
    return swa_eng_translator(text, max_length=128)[0]["generated_text"]


# ==================================================
# SPACY LANGUAGE DETECTOR (AUTO DOWNLOAD SAFE)
# ==================================================

print("Loading spaCy model...")

try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def get_lang_detector(nlp, name):
    return LanguageDetector()


Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe("language_detector", last=True)


# ==================================================
# CHATBOT CORE LOGIC
# ==================================================

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    p = bow(sentence, words)

    res = model.predict(np.array([p]), verbose=0)[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def getResponse(ints):
    if not ints:
        return "Sorry, I didn't understand that."

    tag = ints[0]["intent"]

    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])


def chatbot_response(msg):
    doc = nlp(msg)
    detected_language = doc._.language["language"]

    if detected_language == "sw":
        msg = translate_text_swa_eng(msg)

    response = getResponse(predict_class(msg))

    if detected_language == "sw":
        response = translate_text_eng_swa(response)

    return response


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)
app.static_folder = "static"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg")
    return chatbot_response(userText)


# ==================================================
# PRODUCTION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
