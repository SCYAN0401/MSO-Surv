###

import streamlit as st
from streamlit_shap import st_shap

import pickle
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import sksurv
import sklearn
from sklearn.preprocessing import OrdinalEncoder

###

rsf = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

X_test_ = pickle.load(open('model/X_test.pkl', 'rb'))
explainer = shap.PermutationExplainer(rsf.predict, X_test_)

times = np.arange(0, 360)
best_cop = 5.827909050252443

###

def recode(Age, T_category, N_category, M_category, Stage, Extent, Grade, Tumor_size, Surgery, Hysterectomy, Chemotherapy, Radiotherapy):
    Age_ = Age
    T_category_ = {'T1': 0, 'T2': 1, 'T3': 2}[T_category]
    N_category_ = {'N0': 0, 'N1': 1}[N_category]
    M_category_ = {'M0': 0, 'M1': 1}[M_category]
    Stage_ = {'I': 0, 'II': 1, 'III': 2, 'IV': 2}[Stage]
    Extent_ = {'CTO': 0,'DM/PE': 1}[Extent]
    Grade_ = {'D': 0,'PD/UD': 1}[Grade]
    Tumor_size_ = Tumor_size
    Hysterectomy_ = {'No': 0, 'Yes': 1}[Hysterectomy]
    Chemotherapy_ = {'No/Unknown': 0, 'Yes': 1}[Chemotherapy]
    
    Surgery_BSO = True if Surgery == 'BSO' else False
    Surgery_No = True if Surgery == 'No' else False
    Surgery_PR = True if Surgery == 'PR' else False
    Surgery_SOwO = True if Surgery == 'SOwO' else False
    Surgery_USO = True if Surgery == 'USO' else False
    
    Radiotherapy_EBRT = True if Radiotherapy == 'EBRT' else False
    Radiotherapy_NoUnknown = True if Radiotherapy == 'No/Unknown' else False
    Radiotherapy_RAI = True if Radiotherapy == 'RAI' else False
    

    X_test = pd.DataFrame([Age_, T_category_, N_category_, M_category_, Stage_, Extent_, Grade_, Tumor_size_, 
                           Hysterectomy_, Chemotherapy_, 
                           Surgery_BSO, Surgery_No, Surgery_PR, Surgery_SOwO, Surgery_USO,                         
                           Radiotherapy_EBRT, Radiotherapy_NoUnknown, Radiotherapy_RAI]).transpose()
    X_test.columns = ['Age','T category','N category','M category','AJCC stage','Extent','Grade','Tumor size',
                      'Hysterectomy','Chemotherapy',
                      'Surgery_BSO', 'Surgery_No', 'Surgery_PR', 'Surgery_SOwO', 'Surgery_USO',
                      'Radiotherapy_EBRT', 'Radiotherapy_No/Unknown', 'Radiotherapy_RAI']    
    return X_test

def plot_personalized_predictions(estimator, X, times, best_cop, ax = None):
    rs = estimator.predict(X)
    if rs > best_cop:
        color_ = '#ff7f0e'
        plt.text(0, 0.175, 'High-risk group', color=color_, weight='bold', fontsize = 7)
    else:
        color_ = '#1f77b4'
        plt.text(0, 0.175, 'Low-risk group', color=color_, weight='bold', fontsize = 7)       
    pred_surv = estimator.predict_survival_function(X)
    for surv_func in pred_surv:    
        plt.step(times, surv_func(times), where="post", color=color_)
    plt.xticks(np.arange(0,np.max(times)+30,60), fontsize = 7)
    plt.yticks(fontsize = 7)
    plt.ylim(-0.05, 1.05)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$", fontsize = 7)
    plt.xlabel("time $t$ (months)", fontsize = 7)
    plt.text(0, 0, f'Survival probability\n5-year: {surv_func(60):.1%}\n10-year: {surv_func(120):.1%}', fontsize = 7)
    if ax is None:
        ax = plt.gca()
    return ax

####
def main():
    
    st.set_page_config(layout="wide")
    st.title('MSO-Surv')
    col1, col2 = st.columns(2)
    
    with col1:
        

        st.write('A RSF-based prediction model to estimate the survival probability and stratify the risk for patients with malignant struma ovarii (MSO).\
            MSO-Surv was trained on 120 patients from Surveillance, Epidemiology, and End Results (SEER) database and validated with 194 patients from previous case reports.\
                This prediction model has been developed and validated solely for scientific research purposes and has not been evaluated prospectively or approved for clinical use.')
    
        st.divider()
        col1_, col2_ = st.columns(2)
        
        with col1_:
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
                             captions=['Confined to ovary', 'Distant metastasis/Peritoneal extension'])
            
            Grade = st.radio("**Grade**",
                             ['D','PD/UD'],
                             captions=['Differentiated', 'Poorly differentiated/Undifferentiated'])
    
        with col2_:       
            Tumor_size = st.slider('**Tumor size (mm)**', 
                                   min_value = 1,
                                   max_value = 250)
        
            Surgery = st.radio("**Surgery for the primary tumor**",
                               ["USO", "BSO", "SOwO", "PR", 'No'],
                               captions = ["Unilateral salpingo-oophorectomy", "Bilateral salpingo-oophorectomy", "Salpingo-oophorectomy with omentectomy", 
                                           "Partial resection, such as cystectomy", 'No surgery performed'])
        
            Hysterectomy = st.radio("**Hysterectomy**",
                                    ['No', 'Yes'])
        
            Chemotherapy = st.radio("**Chemotherapy**",
                                    ['No/Unknown', 'Yes'])
            
            Radiotherapy = st.radio("**Radiotherapy**",
                                    ['No/Unknown', 'RAI', 'EBRT'])
    ####
            
            Surgery_PR = 'Yes' if Surgery == 'PR' else 'No'
            Surgery_USO = 'Yes' if Surgery == 'USO' else 'No'
            Radiotherapy_RAI = {'RAI': 'Yes', 'EBRT': 'No', 'No/Unknown': 'No/Unknown'}[Radiotherapy]
            Radiotherapy_EBRT = {'EBRT': 'Yes', 'RAI': 'No', 'No/Unknown': 'No/Unknown'}[Radiotherapy]
            
    ####
        
            if "disabled" not in st.session_state:
                st.session_state['disabled'] = False
    
            st.checkbox('**I understand MSO-Surv is solely for scientific research purposes.**',
                        key="disabled")
        
            if st.button("**Predict**",
                         disabled=operator.not_(st.session_state.disabled)):
                
                
                X_test = recode(Age, T_category, N_category, M_category, Stage, Extent, Grade, Tumor_size, Surgery, Hysterectomy, Chemotherapy, Radiotherapy)
                X_test_scale = scaler.transform(X_test)
                X_test_scale = pd.DataFrame(X_test_scale, columns = X_test.columns)             
                
                X_test_final = X_test_scale[['Age', 'Extent', 'N category', 'Hysterectomy', 'Surgery_PR', 'Chemotherapy', 'M category', 
                                             'Radiotherapy_RAI', 'Surgery_USO', 'Tumor size', 'Radiotherapy_EBRT', 'Grade', 'AJCC stage']]

                ylabels = [
                    str(Age) + ' years' + ' = ' + 'Age',
                    str(Extent) + ' = ' + 'Extent', 
                    str(N_category) + ' = ' + 'N category', 
                    str(Hysterectomy) + ' = ' + 'Hysterectomy', 
                    str(Surgery_PR) + ' = ' + 'Surgery_PR', 
                    str(Chemotherapy) + ' = ' + 'Chemotherapy', 
                    str(M_category) + ' = ' + 'M category', 
                    str(Radiotherapy_RAI) + ' = ' + 'Radiotherapy_RAI', 
                    str(Surgery_USO) + ' = ' + 'Surgery_USO', 
                    str(Tumor_size) + ' mm' + ' = ' + 'Tumor size', 
                    str(Radiotherapy_EBRT) + ' = ' + 'Radiotherapy_EBRT', 
                    str(Grade) + ' = ' + 'Grade', 
                    str(Stage) + ' = ' + 'AJCC stage'
                ]
                             
                with col2:
                    explanation = explainer(X_test_final)



                    combine_list = list(zip(
                        explanation[0].feature_names,
                        np.abs(explanation[0].values),
                        ylabels))
                    
                    sorted_lists = sorted(combine_list, key = lambda x: x[1], reverse = False)
                    sorted_ylabels = [item[2] for item in sorted_lists]
                    sorted_ylabels

                    st.write('SHAP plot')
                    figure = shap.plots.waterfall(explanation[0], max_display=18, show = False)
                    ax_ = figure.get_axes()[0]
                    ax_.set_yticks(np.arange(0, 13, 1))
                    ax_.set_yticklabels(sorted_ylabels)
                    figure = ax_.get_figure()
                    
                    st_shap(figure, width=1000, height=400)
                    
                    ax = plot_personalized_predictions(rsf, X_test_final, times, best_cop)
                    fig = ax.get_figure()
                    fig.set_size_inches(5,3)
                    st.write('KM plot')
                    st.pyplot(fig, use_container_width = False)
                              
if __name__=='__main__':
    main()
