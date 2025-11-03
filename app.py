import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import numpy as np

with open('tokenizer.pickle','rb') as handel:
    tokenizer = pickle.load(handel)



class Textpreproccessed:
    def __init__(self):
        self.max_features = 10000
        self.max_len = 500
    def preproccessed(self,text):
        """
        proccessed text input in one hot and create same length using padd sequesnce
        """
        text = text.lower()
        text = text.strip()
        seq = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(seq,maxlen=self.max_len)
        return padded_seq

class SentimentPredictor:
    def __init__(self):
        self.model = load_model('sentiment.h5')
        self.preproccessed = Textpreproccessed()

    def predict_sentiment(self,text)->tuple:
        """
        predict the sentiment as per text
        """
        text = self.preproccessed.preproccessed(text)
        prediction = self.model.predict(text)
        pred_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        return pred_class,confidence
    

st.title('💬 Sentiment Analysis App')
st.write("Analyze the **sentiment** of your text using an RNN neural network.")

user_input = st.text_area("Enter your sentiment here:", placeholder="Type something...")


if st.button("Analyze"):
    if user_input.strip():
        predictor = SentimentPredictor()
        label, score = predictor.predict_sentiment(user_input)
        classes = {0:"Negative",1:"Neutral",2:"Positive"}
        st.markdown(f"### Prediction: {classes[label]}")
        st.progress(float(score))
        st.write(f"**Confidence:** {score:.2f}")
    else:
        st.warning("Please enter some text first.")






