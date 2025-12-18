# Mental Health Chatbot

A conversational AI chatbot designed to provide mental health support through natural language interaction.  
This chatbot understands and responds to user inputs in  **English**, leveraging language detection and translation models for seamless multilingual support.


## Overview

The Mental Health Chatbot uses a combination of deep learning and natural language processing techniques to:

- Detect the user’s language 
- Translate messages between English  when needed
- Classify user intents using a trained Keras model
- Provide appropriate mental health-related responses based on the detected intent


## Key Features

- **Multilingual Support:** Automatically detects whether input is in English .
- **Neural Machine Translation:** Uses Hugging Face Transformer models to translate between English .
- **Intent Classification:** Classifies user messages into predefined mental health-related intents using a Keras deep learning model.
- **Easy-to-use Web Interface:** Powered by Flask, accessible through a web browser.
- **Customizable Responses:** Configurable responses stored in a JSON file for easy updates and improvements.


## How It Works

1. User sends a message via the web interface.
2. The chatbot detects the language using spaCy’s language detector.
3. The Keras model predicts the intent of the message.
4. The chatbot selects an appropriate response.
5. The chatbot sends the response to the user.


##**Technologies Used**

Python 3.10+
Flask – Web framework
TensorFlow/Keras – Deep learning for intent classification
spaCy – Language detection
NLTK – Text preprocessing
NumPy – Numerical operations


**Chatbot Interface**
<img width="1303" height="668" alt="chatbot screenshot" src="https://github.com/user-attachments/assets/444888ee-91a2-4b9d-9107-caea24f8e5fb" />


  
