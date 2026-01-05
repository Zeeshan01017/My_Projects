import streamlit as st
import pickle
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Load model and vectorizer (dummy placeholders)
tfidf = pickle.load(open('Machine_Learning_Projects/sms_spam_classification/vectorizer.pkl','rb'))
model = pickle.load(open('Machine_Learning_Projects/sms_spam_classification/model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the message")
if input_sms:
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
