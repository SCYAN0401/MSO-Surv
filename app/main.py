###

import streamlit as st
import pickle
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sksurv
import sklearn
from sklearn.preprocessing import OrdinalEncoder

###

rsf = pickle.load(open('model/model.pkl', 'rb'))
# imputer = pickle.load(open('model/imputer.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

###

def onehot_encoder(X, va_name, feat_name, new_col_name):
    X_oh_ = []
    for feat in feat_name:
        if X.loc[0, va_name] == feat:
            X_oh_.append(True)
        else:
            X_oh_.append(False)
    X_oh = pd.DataFrame(X_oh_).transpose()
    X_oh.columns = new_col_name    
    rest_names = [name for name in X.columns.tolist() if name != va_name]
    Xt = pd.concat([X[rest_names], X_oh], axis=1)
    return Xt

def ordinal_encoder(X, va_name, cat_name_list):
    va_str = X.loc[:, va_name].astype(object).values[:, np.newaxis]
    va_num = OrdinalEncoder(categories=[cat_name_list], handle_unknown = 'use_encoded_value',  unknown_value = np.nan).fit_transform(va_str)
    Xt = X.drop(va_name, axis=1)
    Xt.loc[:, va_name] = va_num
    Xt = Xt.loc[:, X.columns.tolist()]
    return Xt

def ENCODER(X):
    X = ordinal_encoder(X, 'T category', ['T1','T2','T3'])
    X = ordinal_encoder(X, 'N category', ['N0','N1'])
    X = ordinal_encoder(X, 'M category', ['M0','M1'])
    X = ordinal_encoder(X, 'AJCC stage', ['I','II','III','IV'])
    X = ordinal_encoder(X, 'Extent', ['CTO','DM/PE'])
    X = ordinal_encoder(X, 'Grade', ['G1','>G1'])
    X = ordinal_encoder(X, 'Hysterectomy', ['No','Yes'])
    X = ordinal_encoder(X, 'Chemotherapy', ['No/Unknown','Yes'])
    
    X = onehot_encoder(X, 'Surgery', ["USO", "BSO", "SOwO", "PR", 'No'], ['Surgery_USO', 'Surgery_BSO', 'Surgery_SOwO', 'Surgery_PR', 'Surgery_No'])
    X = onehot_encoder(X, 'Radiotherapy', ['No/Unknown','RAI', 'EBRT'], ['Radiotherapy_No/Unknown','Radiotherapy_RAI', 'Radiotherapy_EBRT'])
    return X

def preprocessor_test(X_test, encoder, scaler_):
    X_test_encode = encoder(X_test)
    X_test_scale = scaler_.transform(X_test_encode)
    X_test_scale = pd.DataFrame(X_test_scale, columns = X_test_encode.columns)
    
    return X_test_scale

def plot_personalized_predictions(estimator, X, times, best_cop, ax = None):
    rs = estimator.predict(X)
    if rs > best_cop:
        color_ = '#ff7f0e'
        plt.text(0, 0.175, 'High-risk group', color=color_, weight='bold')
    else:
        color_ = '#1f77b4'
        plt.text(0, 0.175, 'Low-risk group', color=color_, weight='bold')       
    pred_surv = estimator.predict_survival_function(X)
    for surv_func in pred_surv:    
        plt.step(times, surv_func(times), where="post", color=color_)
    plt.xticks(np.arange(0,np.max(times)+30,60))
    plt.ylim(-0.05, 1.05)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$ (months)")
    plt.text(0, 0, f'Survival probability\n5-year: {surv_func(60):.1%}\n10-year: {surv_func(120):.1%}')
    if ax is None:
        ax = plt.gca()
    return ax

####

def main():
    
    st.title('MSO-Surv')

    st.write('A RSF-based prediction model to estimate the survival probability and stratify the risk for patients with malignant struma ovarii (MSO).\
        MSO-Surv was trained on 120 patients from Surveillance, Epidemiology, and End Results (SEER) database and validated with 194 patients from previous case reports.\
            This prediction model has been developed and validated solely for scientific research purposes and has not been evaluated prospectively or approved for clinical use.')

    st.divider()

    Age = st.slider('**Age (years)**',
                    min_value = 11, 
                    max_value = 82)

    T_category = st.radio("**T category (AJCC for ovarian cancer)**",
                          ["T1", "T2", 'T3'])

    N_category = st.radio("**N category (AJCC for ovarian cancer)**",
                          ["N0", "N1"])
    
    M_category = st.radio("**M category (AJCC for ovarian cancer)**",
                          ["M0", "M1"])
    
    Stage = st.radio("**Stage (AJCC for ovarian cancer)**",
                     ['I','II','III','IV'] )
    
    Extent = st.radio("**Extent of tumor**",
                     ['CTO','DM/PE'],
                     captions=['confined to ovary', 'distant metastasis/peritoneal extension'])
    
    Grade = st.radio("**Grade**",
                     ['G1','>G1'],
                     captions=['Differentiated', 'Poorly differentiated/Undifferentiated'])
       
    Tumor_size = st.slider('**Tumor size (mm)**', 
                           min_value = 1,
                           max_value = 250)

    Surgery = st.radio("**Surgery for the primary tumor**",
                       ["USO", "BSO", "SOwO", "PR", 'No'],
                       captions = ["Unilateral salpingo-oophorectomy", "Bilateral salpingo-oophorectomy", "Salpingo-oophorectomy with omentectomy", "Partial resection, such as cystectomy", 'No surgery performed'])

    Hysterectomy = st.radio("**Hysterectomy**",
                            ['No','Yes'])

    Chemotherapy = st.radio("**Chemotherapy**",
                            ['No/Unknown','Yes'])
    
    Radiotherapy = st.radio("**Radiotherapy**",
                            ['No/Unknown','RAI', 'EBRT'])

    st.divider()

####
    
    if "disabled" not in st.session_state:
        st.session_state['disabled'] = False
    
    st.checkbox('**I understand MSO-Surv is solely for scientific research purposes.**',
                key="disabled")
    
    if st.button("**Predict**",
                 disabled=operator.not_(st.session_state.disabled)):
        
        
        X_test = pd.DataFrame([Age, T_category, N_category, M_category, Stage, Extent, Grade, Tumor_size, Surgery, Hysterectomy, Chemotherapy, Radiotherapy]).transpose()
        X_test.columns = ['Age','T category','N category','M category','AJCC stage','Extent','Grade','Tumor size','Surgery','Hysterectomy','Chemotherapy','Radiotherapy']

        X_test_encode = ENCODER(X_test)
        X_test_scale = scaler.transform(X_test_encode)
        X_test_scale = pd.DataFrame(X_test_scale, columns = X_test_encode.columns)             
        X_test_final = X_test_scale

        times = np.arange(0, 360)
        best_cop = 5.827909050252443
        
        fig = plot_personalized_predictions(rsf, 
                                            X_test_final, 
                                            times, 
                                            best_cop)
        
        st.pyplot(fig)
            
if __name__=='__main__':
    main()
