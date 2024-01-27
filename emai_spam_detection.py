import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)


tfidf_path = r'C:\Users\Thummala Babitha\machine-learning\vectorizer.pkl'
with open(tfidf_path, 'rb') as file:
    tfidf = pickle.load(file)


model_path = r'C:\Users\Thummala Babitha\machine-learning\model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title("Email/sms classifier:")

input_sms = st.text_input("Enter the message:")
if st.button('predict'):    
    transform_sms = transform(input_sms)

    
    vector_input = tfidf.transform([transform_sms])

    
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
