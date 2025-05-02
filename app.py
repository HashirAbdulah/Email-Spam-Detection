import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
import os

# Ensure necessary NLTK data is downloaded (only once)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(nltk_data_path, "tokenizers", "punkt")):
    nltk.download('punkt')
if not os.path.exists(os.path.join(nltk_data_path, "corpora", "stopwords")):
    nltk.download('stopwords')

# Initialize NLTK tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.title("📧 Email/SMS Spam Classifier")

# Text input field using session state
input_sms = st.text_area('Enter the Message', key='input_sms')

# Predict button
if st.button('Predict'):
    if input_sms.strip() != "":
        with st.spinner('Classifying...'):
            # Pre-process the input
            transform_sms = transform_text(input_sms)
            # Vectorize
            vector_input = tfidf.transform([transform_sms]).toarray()
            # Predict
            result = model.predict(vector_input)[0]
            prob = model.predict_proba(vector_input)[0]

            # Display result
            st.markdown("### Result:")
            st.success("🚫 Spam" if result == 1 else "✅ Not Spam")
            st.markdown(f"**📛 Spam Probability:** `{prob[1]:.2%}`")
            st.markdown(f"**📩 Ham Probability:** `{prob[0]:.2%}`")
    else:
        st.warning("Please enter a message to classify.")
