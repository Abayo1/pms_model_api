# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:01:50 2023

@author: Abayo
"""
# loading required libraries
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import pickle
import json

#loading an instance of fastapi

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    
    )

# mentioning the input formats we need

class modelTemp_input(BaseModel):
    body_temperature: float
class modelHeartRate_input(BaseModel):
    heart_rate: float
class modelRespiratoryRate_input(BaseModel):
    respiratory_rate: float
class modelSPO2_input(BaseModel):
    SPO2: float
class modelSystolicPressure_input(BaseModel):
    systolic_blood_pressure: float
class modelDiastolicPressure_input(BaseModel):
    diastolic_blood_pressure: float
    
    
# loading the saved models
bt_model = pickle.load(open('modelSVR_BT.0.1.0.sav','rb'))
hr_model = pickle.load(open('modelSVR_HR.0.1.0.sav','rb'))
rr_model = pickle.load(open('modelSVR_RR.0.1.0.sav','rb'))
spo2_model = pickle.load(open('modelSVR_SPO2.0.1.0.sav','rb'))
sys_model = pickle.load(open('modelSVR_SBP.0.1.0.sav','rb'))
dys_model = pickle.load(open('modelSVR_DBP.0.1.0.sav','rb'))

# Creating an APIs

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post('/body_temperature_pred')

def body_temperature_pred(input_parameters: modelTemp_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    temp = input_dictionary['body_temperature']

    input_list1 = [temp]
    #reshape input value
    input_list1 = np.reshape(input_list1,(-1,1))
    #transform input values
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_list1 = scaler.fit_transform(input_list1)
    #make prediction
    predict = bt_model.predict(input_list1)
    predict = np.reshape(predict,(-1,1))
    predictionBT = scaler.inverse_transform(predict)
    predictionBT = print(predictionBT)
    return {predictionBT}
    

@app.post('/heart_rate_pred')

def heart_rate_pred(input_parameters: modelHeartRate_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    hr = input_dictionary['heart_rate']

    input_list2 = [hr]
    #reshape input value
    input_list2 = np.reshape(input_list2,(-1,1))
    #transform input values
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_list2 = scaler.fit_transform(input_list2)
    #make prediction
    predict = bt_model.predict(input_list2)
    predict = np.reshape(predict,(-1,1))
    predictionHR = scaler.inverse_transform(predict)
    predictionHR = print(predictionHR)
    return {predictionHR}

@app.post('/respiratory_rate_pred')

def respiratory_rate_pred(input_parameters: modelRespiratoryRate_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    rr = input_dictionary['respiratory_rate']

    input_list3 = [rr]
    #reshape input value
    input_list3 = np.reshape(input_list3,(-1,1))
    #transform input values
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_list3 = scaler.fit_transform(input_list3)
    #make prediction
    predict = rr_model.predict(input_list3)
    predict = np.reshape(predict,(-1,1))
    predictionRR = scaler.inverse_transform(predict)
    predictionRR = print(predictionRR)
    return {predictionRR}

@app.post('/spo2_pred')

def spo2_pred(input_parameters: modelSPO2_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    spo2 = input_dictionary['spo2']

    input_list4 = [spo2]
    #reshape input value
    input_list4 = np.reshape(input_list4,(-1,1))
    #transform input values
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_list4 = scaler.fit_transform(input_list4)
    #make prediction
    predict = spo2_model.predict(input_list4)
    predict = np.reshape(predict,(-1,1))
    predictionSPO2 = scaler.inverse_transform(predict)
    predictionSPO2 = print(predictionSPO2)
    return {predictionSPO2}

@app.post('/systolic_pred')

def sys_pred(input_parameters: modelSystolicPressure_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    sys = input_dictionary['systolic_blood_pressure']

    input_list5 = [sys]
    #reshape input value
    input_list5 = np.reshape(input_list5,(-1,1))
    #transform input values
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_list5 = scaler.fit_transform(input_list5)
    #make prediction
    predict = sys_model.predict(input_list5)
    predict = np.reshape(predict,(-1,1))
    predictionSYS = scaler.inverse_transform(predict)
    predictionSYS = print(predictionSYS)
    return {predictionSYS}

@app.post('/diastolic_pred')

def dys_pred(input_parameters: modelDiastolicPressure_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    dias = input_dictionary['diastolic_blood_pressure']

    input_list6 = [dias]
    #reshape input value
    input_list6 = np.reshape(input_list6,(-1,1))
    #transform input values
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_list6 = scaler.fit_transform(input_list6)
    #make prediction
    predict = dys_model.predict(input_list6)
    predict = np.reshape(predict,(-1,1))
    predictionDIAS = scaler.inverse_transform(predict)
    predictionDIAS = print(predictionDIAS)
    return {predictionDIAS}
