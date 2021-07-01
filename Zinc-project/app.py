import streamlit as st
from pycaret.regression import *
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('Zinc')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image_1 = Image.open('AIideagram.jpg')
    image_2 = Image.open('image.jpg')
    image_3 = Image.open('Zinc.jpg')
    image_4 = Image.open('Zinc-Periodic.jpg')

    st.image(image_1,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('The goal of this sharing is to try to use AutoML Pycarets powerful tool for predicting the price of zinc based on a number of chosen parameters. Nine of main ore product and energy parameters were chosen, namely: Gold, Iron-Ore, Lead, Silver, Zinc, rock phosphate, Crude Oil, Natural Gas, Rubber. I added to this 6 agricultural source parameters that could be useful and whose price fluctuation could be influential, Rice, Maize, Coffee, Tea, Cotton and Plywood. The data represents a 30-years history of the official selling prices of each parameter. The dataset contains 14 features with 343 observations (between May 1991 and May 2021). Data have been divised on three parts 95% of train and test (respectively 70% and 30%) and 5% as an unseen data for the last validation of the best choosen model.')
    st.sidebar.success('Forcasting & Futuristics')
    
    st.sidebar.image(image_2)

    st.title("Zinc price prediction using PyCaret")

    if add_selectbox == 'Online':

        Gold=st.number_input('Gold', min_value=300, max_value=2500, value=1200)
        Silver=st.number_input('Silver', min_value=3, max_value=50, value=20)
        Iron-Ore=st.number_input('Iron-Ore', min_value=20, max_value=300, value=25)
        Crude_Oil=st.number_input('Crude_Oil', min_value=10, max_value=115, value=25)
        rockphos=st.number_input('rockphos', min_value=20, max_value=600, value=100)
        

        output=""

        input_dict = {'Gold' : Gold, 'Silver' : Silver, 'Iron-Ore' : Iron-Ore, 'Crude_Oil' : Crude_Oil, 'rockphos' : rockphos}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = 'The Zinc Price Prediction is :' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
