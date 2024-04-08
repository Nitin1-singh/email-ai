import streamlit as st
import pickle as pk
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
stem = PorterStemmer()
nltk.download('stopwords')

def tranform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)
  text =  y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    y.append(stem.stem(i))
  return " ".join(y)

cv = pk.load(open("./model/cv.pkl","rb"))
model = pk.load(open("./model/mnb.pkl","rb"))
st.title("Email/SMS Spam Classifier","rb")
input_sms = st.text_area("Enter your input")


if st.button("Predict"):
  tranform_input = tranform_text(input_sms)
  vector_input = cv.transform([tranform_input])
  result = model.predict(vector_input)
  if result[0] == 0:
    st.header("Not Spam")
  else:
    st.header("Spam")



