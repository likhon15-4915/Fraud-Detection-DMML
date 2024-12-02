import joblib
import streamlit as st

model = joblib.load('randomforest.joblib')
minmaxscalar = joblib.load('minmaxscalar.joblib')
labelencoder = joblib.load('labelencoder.joblib')

step = st.number_input("step")
type = st.number_input("type")
amount = st.number_input("amount")
oldbalanceOrg = st.number_input("oldbalanceOrg")
newbalanceOrig= st.number_input("newbalanceOrig")
oldbalanceDest = st.number_input("oldbalanceDest")
newbalanceDest= st.number_input("newbalanceDest")
def predict_class():
    values = values = [[step, type, amount,oldbalanceOrg, newbalanceOrig, oldbalanceDest,newbalanceDest]]
    values = minmaxscalar.transform(values)
    predicted_class = model.predict(values)
    class_name = labelencoder.inverse_transform(predicted_class)[0]
    st.write(class_name)

st.button('Predict', on_click=predict_class)