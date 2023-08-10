from fastapi import FastAPI
import pickle
import json
import numpy as np

app = FastAPI()

@app.get('/')

def home():
    return {'text':'PMS Prediction model'}

@app.post('/temp_pred')
def temp_pred(temp:str):
    model = pickle.load(open(r'E:/New folder/machine_learning/PMS_model_deploy/models/modelSVR_BT.0.2.0.sav','rb'))           
    predict = model.predict([[temp]])
    output = round(predict[0],3)
    return {output}

@app.post('/heart_rate_pred')
def heart_rate_pred(heart_rate:str):
    model = pickle.load(open(r'E:/New folder/machine_learning/PMS_model_deploy/models/modelSVR_HR.0.2.0.sav','rb'))
    predict = model.predict([[heart_rate]])
    output = round(predict[0],3)
    return {output}

@app.post('/respiratory_rate_pred')
def respiratory_rate_pred(respiratory_rate:str):
    model = pickle.load(open(r'E:/New folder/machine_learning/PMS_model_deploy/models/modelSVR_RR.0.2.0.sav','rb'))
    predict = model.predict([[respiratory_rate]])
    output = round(predict[0],3)
    return {output}

@app.post('/spo2_pred')
def spo2_pred(spo2:str):
    model = pickle.load(open(r'E:/New folder/machine_learning/PMS_model_deploy/models/modelSVR_SPO2.0.2.0.sav','rb'))
    predict = model.predict([[spo2]])
    output = round(predict[0],3)
    return {output}

@app.post('/sys_pred')
def sys_pred(sys:str):
    model = pickle.load(open(r'E:/New folder/machine_learning/PMS_model_deploy/models/modelSVR_SYS.0.2.0.sav','rb'))
    predict = model.predict([[sys]])
    output = round(predict[0],3)
    return {output}

@app.post('/dys_pred')
def dys_pred(dys:str):
    model = pickle.load(open(r'E:/New folder/machine_learning/PMS_model_deploy/models/modelSVR_DYS.0.2.0.sav','rb'))
    predict = model.predict([[dys]])
    output = round(predict[0],3)
    return {output}