import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


st.write('### Preadicion of Antracene and Phenentarine removal efficiency in PAH Contaminated Soil using Hybrid ozonation-ultrasonication method ')
XGBR_ANT=joblib.load('XGBR_ANT.sav')
GBR_ANT=joblib.load('GBR_ANT.sav')
RFR_ANT=joblib.load('RFR_ANT.sav')
MLR_ANT=joblib.load('MLR_ANT.sav')

XGBR_PHE=joblib.load('XGBR_PHE.sav')
GBR_PHE=joblib.load('GBR_PHE.sav')
RFR_PHE=joblib.load('RFR_PHE.sav')
MLR_PHE=joblib.load('MLR_PHE.sav')


st.sidebar.title('Inpute Features value')
st.sidebar.write('please determine the inpute features value')

Time_value=st.sidebar.slider('Time(min)', min_value=0.0, max_value=10.0, value=1.0, step=0.25)
Wv_value=st.sidebar.slider('Water content(ml)', min_value=0.0, max_value=400.0, value=100.0, step=5.0)
ANTCons_value=st.sidebar.slider('Antracene Concentration(mg/kg soil)', min_value=7.0, max_value=520.0, value=100.0, step=1.0)
PHECons_value=st.sidebar.slider('Phenentarine Concentration(mg/kg soil', min_value=7.0, max_value=520.0, value=100.0, step=1.0)
O3_value=st.sidebar.slider('Ozone  (gr/h)', min_value=0.0, max_value=6.0, value=1.6, step=0.1)
USP_value=st.sidebar.slider('Ultrasonic Power(watt)', min_value=0.0, max_value=400.0, value=110.0, step=10.0)
Sand_value=st.sidebar.slider('Sand(%)', min_value=5.0, max_value=50.0, value=13.5, step=0.5)
Silt_value=st.sidebar.slider('Silt(%)', min_value=20.0, max_value=70.0, value=100.0-Sand_value-20.0, step=0.5)
Clay_value=100.0-Sand_value-Silt_value
st.sidebar.write('Clay content is : ' + str(Clay_value) + '%')

column_names = ['Time', 'Wv', 'PHACons', 'O3', 'USP', 'Sand', 'Silt', 'Clay']
X = pd.DataFrame(columns=column_names)
X= X.append({'Time':   Time_value,
                'Wv': Wv_value,
                'PHACons':     ANTCons_value,
                'O3':   O3_value,
                'USP':    USP_value,
                'Sand':  Sand_value,
                'Silt': Silt_value,
                'Clay':   Clay_value,
                },
               ignore_index=True)

ANT_pred_XGBR = XGBR_ANT.predict(X).round(2)
ANT_pred_GBR = GBR_ANT.predict(X).round(2)
ANT_pred_RFR = RFR_ANT.predict(X).round(2)
ANT_pred_MLR = MLR_ANT.predict(X).round(2)

st.write('--------------------------------------------------------------------------------------------')
st.write('### Antracene Prediction')
st.write(X)
st.write('Anteracene Removal efficiency based on XGBR model is about (the most reliable) : ' + str(ANT_pred_XGBR) + ' %')
st.write('Anteracene Removal efficiency based on GBR model is about  : ' + str(ANT_pred_GBR) + ' %')
st.write('Anteracene Removal efficiency based on RFR model is about  : ' + str(ANT_pred_RFR) + ' %')
st.write('Anteracene Removal efficiency based on MLR model is about  : ' + str(ANT_pred_MLR) + ' %')

X = pd.DataFrame(columns=column_names)
X= X.append({'Time':   Time_value,
                'Wv': Wv_value,
                'PHACons':     PHECons_value,
                'O3':   O3_value,
                'USP':    USP_value,
                'Sand':  Sand_value,
                'Silt': Silt_value,
                'Clay':   Clay_value,
                },
               ignore_index=True)

PHE_pred_XGBR = XGBR_PHE.predict(X).round(2)
PHE_pred_GBR = GBR_PHE.predict(X).round(2)
PHE_pred_RFR = RFR_PHE.predict(X).round(2)
PHE_pred_MLR = MLR_PHE.predict(X).round(2)

st.write('--------------------------------------------------------------------------------------------')
st.write('### Phenentarine Prediction')
st.write(X)
st.write('Phenentarine Removal efficiency based on XGBR model is about (the most reliable) : ' + str(PHE_pred_XGBR) + ' %')
st.write('Phenentarine Removal efficiency based on GBR model is about  : ' + str(PHE_pred_GBR) + ' %')
st.write('Phenentarine Removal efficiency based on RFR model is about  : ' + str(PHE_pred_RFR) + ' %')
st.write('Phenentarine Removal efficiency based on MLR model is about  : ' + str(PHE_pred_MLR) + ' %')

st.write('--------------------------------------------------------------------------------------------')
st.write('XGBR: eXtreem Gradient Bossting Regressor GBR: Gradient Boosting Regressor ')
st.write('GBR:  Gradient Boosting Regressor ')
st.write('RFR: Random Forest Regressor  ')
st.write('MLR: MultiLinear Regressor ')
st.write('--------------------------------------------------------------------------------------------')
st.write('to see more details about the models accuracy and reliability please read the article : ')
st.write('### Intelligent Models as Novel Tools for Optimizing Ultrasonication-Ozonation Technique in PAH-contaminated Soil Remediation ')
st.write('doi; xxx-xxxxxxxxxxx ')


