import streamlit as st
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import pandas as pd

model = load_model('model.h5')

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def predict_stress_anxiety(stringInput):
    datasetfinal = pd.read_csv("dataset_modedlled.csv")
    datasetfinal.rename(columns = {'Comments Text':'text'},inplace = True)
    for i in range(len(datasetfinal)):
        datasetfinal.at[i,'text'] = remove_urls(datasetfinal.iloc[i]['text'])
    data = datasetfinal['text'].values.tolist()
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data = [re.sub(",", "", sent) for sent in data]
    data = [sent.lower() for sent in data]
    data = [sent.replace('.', '') for sent in data]
    tweets = np.array(data)
    max_words = 20000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(tweets)
    stringInput.replace(r'[^\x00-\x7F]+','')
    stringInput = remove_urls(stringInput)
    data = [stringInput]
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data = np.array(data)
    sequences = tokenizer.texts_to_sequences(data)
    tweet = pad_sequences(sequences, maxlen=400)
    print(tweet)
    prediction = model.predict(tweet)
    prediction = np.around(prediction, decimals=0)
    if prediction == 1.0:
        prediction = "Yes the statement shows stress and anxiety"
    else:
        prediction = "No the statement does not show Stress and Anxiety"
    return prediction

def main():
    st.title("Stress Anxiety Detector")
    input = st.text_input("Enter a Comment/Tweet","Type Here")
    result = ''
    if st.button("Predict"):
        result = predict_stress_anxiety(input)
    st.success('Final Verdict: {}'.format(result))
    if st.button("About"):
        st.text("This is a Stress and Anxiety Detector.")
        st.text(" Enter a User's Tweet or Facebook Comment \n Or any other post in order to detect presence of stress or Anxiety")

if __name__ == '__main__':
    main()