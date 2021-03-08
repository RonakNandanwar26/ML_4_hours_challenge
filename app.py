
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image


pkl = open("classifier.pickle","rb")
classifier=pickle.load(pkl)


def predict(col1,col2):
    prediction=classifier.predict([[col1,col2]])
    return prediction



def main():
    st.title("Classifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Classifier ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    col1 = st.text_input("Col1")
    col2 = st.text_input("Col2")
    result=""
    if st.button("Predict"):
        result=predict(col1,col2)
    st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()
    
    
    
    
