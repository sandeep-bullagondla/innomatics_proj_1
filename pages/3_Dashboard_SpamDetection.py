import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
data = data[["class", "message"]] 

X = np.array(data['message']) 
y = np.array(data['class']) 

cv = CountVectorizer() 
X = cv.fit_transform(X) 

trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.33, random_state = 42) 

clf = MultinomialNB().fit(trainX, trainY) 

import streamlit as st 

st.title("Spam Detection system") 
st.text("In the below text area you can enter any message, it detects the words and displays either 'SPAM' or 'HAM'. If the output is HAM its genuine message else Not.")
def spam_detection(): 
    user = st.text_area("Enter any message or mail:" ) 
    if len(user)<1: 
        st.write(" ") 
    
    else: 
        sample = user 
        data = cv.transform([sample]).toarray()
        a = clf.predict(data) 
        st.title(a)

spam_detection()