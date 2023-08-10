from fastapi import FastAPI
import pickle
import json
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    
    )

@app.get('/')

def home():
    return {'text':'PMS Prediction model'}

@app.post('/temp_pred')
def temp_pred(temp:str):
    model = pickle.load(open('modelSVR_BT.0.2.0.sav','rb'))           
    predict = model.predict([[temp]])
    output = round(predict[0],3)
    return {output}

@app.post('/heart_rate_pred')
def heart_rate_pred(heart_rate:str):
    model = pickle.load(open('modelSVR_HR.0.2.0.sav','rb'))
    predict = model.predict([[heart_rate]])
    output = round(predict[0],3)
    return {output}

@app.post('/respiratory_rate_pred')
def respiratory_rate_pred(respiratory_rate:str):
    model = pickle.load(open('modelSVR_RR.0.2.0.sav','rb'))
    predict = model.predict([[respiratory_rate]])
    output = round(predict[0],3)
    return {output}

@app.post('/spo2_pred')
def spo2_pred(spo2:str):
    model = pickle.load(open('modelSVR_SPO2.0.2.0.sav','rb'))
    predict = model.predict([[spo2]])
    output = round(predict[0],3)
    return {output}

@app.post('/sys_pred')
def sys_pred(sys:str):
    model = pickle.load(open('modelSVR_SYS.0.2.0.sav','rb'))
    predict = model.predict([[sys]])
    output = round(predict[0],3)
    return {output}

@app.post('/dys_pred')
def dys_pred(dys:str):
    model = pickle.load(open('modelSVR_DYS.0.2.0.sav','rb'))
    predict = model.predict([[dys]])
    output = round(predict[0],3)
    return {output}
