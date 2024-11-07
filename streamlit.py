import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import warnings

warnings.filterwarnings('ignore')

st.title("Africa Financial Inclusion")

stat={ 'location_type':0, 'cellphone_access':0,
       'household_size':0, 'age_of_respondent':0, 'gender_of_respondent':0, 'relationship_with_head':0,
       'marital_status':0, 'education_level':0, 'job_type':0, 'country_Kenya':0.0,
       'country_Rwanda':0.0, 'country_Tanzania':0.0, 'country_Uganda':0.0}

gender = st.radio(
        'Gender:',
        ('Male','Female'))
if gender:
    if gender=='Male':
        stat['gender_of_respondent']=1
    else:
        stat['gender_of_respondent']=0

age=st.number_input(f"Input Age:")
if age:
    try:
        age=int(age)
        if age >0 and age <120:
            stat['age_of_respondent']=age
        else:
            st.warning("input valid age")
    except:
        st.error("input not valid")


country = st.radio(
        'Choose the Country',
        ('Rwanda' , 'Kenya' , 'Tanzania', 'Uganda'))
if country:
    count='country_'+country
    stat[count]=1.0

location = st.radio(
        'Location Type:',
        ('Rural','Urban'))
if location:
    if location=='Urban':
        stat['location_type']=1
    else:
        stat['location_type']=0


phone = st.radio(
        'Cellphone access:',
        ('Yes','No'))
if phone:
    if phone=='Yes':
        stat['cellphone_access']=1
    else:
        stat['cellphone_access']=0

size=st.number_input(f"Input household size:")
if size:
    try:
        size=int(size)
        if size >= 0 and size < 30:
            stat['household_size']=size
        else:
            st.warning("input valid size")
    except:
        st.error("input not valid")


with open('education.pkl', 'rb') as f:
    education = pickle.load(f)

with open('job.pkl', 'rb') as f:
    job = pickle.load(f)

with open('martial.pkl', 'rb') as f:
    martial = pickle.load(f)

with open('relationship.pkl', 'rb') as f:
    relationship = pickle.load(f)

st.header("Input subscriber statistics")
sta= st.selectbox(
    'Martial Status',
    ('Married/Living together', 'Single/Never Married', 'Widowed', 'Divorced/Seperated'))

if sta:
    stat['marital_status']=int(martial.transform([sta])[0])


relation= st.selectbox(
    'Relationship with head of family',
    ( 'Head of Household','Spouse','Child' , 'Parent' , 'Other relative' , 'Other non-relatives'))

if relation:
    stat['relationship_with_head']=int(relationship.transform([relation])[0])

educa= st.selectbox(
    'Education Level',
    ( 'Primary education' , 'No formal education' , 'Secondary education' , 'Tertiary education' , 'Vocational/Specialised training'))

if relation:
    stat['education_level']=int(education.transform([educa])[0])


job0= st.selectbox(
    'Job Type',
        ('Self employed' , 'Informally employed' , 'Farming and Fishing' , 'Remittance Dependent' , 
         'Other Income' ,'Formally employed Private' , 'No Income' , 'Formally employed Government' , 'Government Dependent'))

if job0:
    stat['job_type']=int(job.transform([job0])[0])




st.text('Summary of Subscriber data:')
df=pd.DataFrame(stat,index=[0])
st.table(df)

model = joblib.load('best_model.joblib')


if st.button("Prediction: Has Account/Not has Account:"):
    result=model.predict(df)
    pred=''
    if result[0]==0: 
        pred='No'
    else:
        pred='Yes'
    st.text(f'This respondent has account? : {pred} ')


