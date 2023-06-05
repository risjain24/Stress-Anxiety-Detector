import streamlit as st
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import pandas as pd

model = load_model('model.h5')
max_words = 20000
tokenizer = Tokenizer(num_words=max_words)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocessing():
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
    tokenizer.fit_on_texts(tweets)

def predict_stress_anxiety(stringInput):
    stringInput.replace(r'[^\x00-\x7F]+','')
    stringInput = remove_urls(stringInput)
    data = [stringInput]
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data = np.array(data)
    sequences = tokenizer.texts_to_sequences(data)
    tweet = pad_sequences(sequences, maxlen=400)
    prediction = model.predict(tweet)
    return prediction

def main():
    preprocessing()
    st.title("Stress Anxiety Detector")
    input = st.text_input("Enter a Comment/Tweet","Type Here")
    result = 0.0
    if st.button("Predict"):
        result = predict_stress_anxiety(input)
        result = result[0][0].astype(float)
        resApprox = np.around(result, decimals=0)
        if resApprox == 1.0:
            st.success('Yes the statement shows signs of stress and anxiety')
            st.text("Confidence Score : " + str(result))
            st.progress(result)
        else:
            st.markdown(
                """
                <div style="background-color: #f05b5b; padding: 10px; border-radius: 5px;">
                <span style="color: white;">No the statement does not show signs of Stress and Anxiety</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            st.text("Confidence Score : " + str(1-result))
            st.progress((1-result))
            
    if st.button("Help"):
        st.text("This is a Stress and Anxiety Detector.")
        st.text(" Enter a User's Tweet or Facebook Comment \n Or any other post in order to detect presence of stress or Anxiety")
    if st.button("About"):
        st.text("Made with ♥ by Students of BVCOE, New Delhi")

if __name__ == '__main__':
    main()