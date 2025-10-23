import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model



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
        proccessed_text = one_hot(text,self.max_features)
        padded_text = tf.keras.utils.pad_sequences([proccessed_text],self.max_len)
        return padded_text
    

class SentimentPredictor:
    def __init__(self):
        self.model = load_model('simple_rnn_imdb.h5')
        self.preproccessed = Textpreproccessed()

    def predict_sentiment(self,text)->tuple:
        """
        predict the sentiment as per text
        """
        text = self.preproccessed.preproccessed(text)
        prediction = self.model.predict(text)
        sentiment = 'Positive' if prediction[0][0]> 0.5 else 'Negative'
        return sentiment,prediction[0][0]
    

st.title('💬 Sentiment Analysis App')
st.write("Analyze the **sentiment** of your text using an RNN neural network.")

user_input = st.text_area("Enter your sentiment here:", placeholder="Type something...")


if st.button("Analyze"):
    if user_input.strip():
        predictor = SentimentPredictor()
        label, score = predictor.predict_sentiment(user_input)
        st.markdown(f"### Prediction: {label}")
        st.progress(score if label == "Positive 😀" else int((1 - score) * 100))
        st.write(f"**Confidence:** {score:.2f}")
    else:
        st.warning("Please enter some text first.")






