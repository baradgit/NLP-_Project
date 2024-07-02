import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import os


from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

script_dir = os.path.dirname(__file__)

# Load the vectorizer and model from the same directory
tfidf_path = os.path.join(script_dir, 'vectorizer.pkl')
model_path = os.path.join(script_dir, 'model.pkl')

tfidf = pickle.load(open(tfidf_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

#tfidf = pickle.load(open('vectorizer.pkl','rb'))
#model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Check if the input contains at least 30 words
    if len(input_sms.split()) < 30:
        st.error("Please enter at least 30 words.")
    else:
        # Preprocess the input text
        transformed_sms = transform_text(input_sms)
        # Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])
        # Predict using the model
        result = model.predict(vector_input)[0]
        # Display the prediction result
        if result == 1:
            st.header("Spam")
        elif result == 0:
            st.header("Not Spam")
