# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('fetal_health.csv')

st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', width = 800)
st.write('Utilize our advanced Machine Learning application to predict fetal health applications')



st.sidebar.header('Fetal Health Features Input')
file = st.sidebar.file_uploader("Upload your data", help='File must be in CSV format', type='csv')

st.sidebar.warning('Ensure your data strictly follows the format outlined below.', icon='⚠️')
st.sidebar.dataframe(df)
alg = st.sidebar.radio('Choose Model for Prediction', options=['Random Forest', 'Decision Tree', 'Ada Boost', 'Soft Voting'])
st.sidebar.info(f'You selected: {alg}', icon=":material/check_circle:")

if alg == 'Random Forest':
    # Load the pre-trained model from the pickle file
    rf_pickle = open('rf_fetal.pickle', 'rb') 
    clf = pickle.load(rf_pickle) 
    rf_pickle.close()
elif alg == 'Decision Tree':
    # Load the pre-trained model from the pickle file
    dt_pickle = open('dt_fetal.pickle', 'rb') 
    clf = pickle.load(dt_pickle) 
    dt_pickle.close()
elif alg == 'Ada Boost':
    # Load the pre-trained model from the pickle file
    ada_pickle = open('ada_fetal.pickle', 'rb') 
    clf = pickle.load(ada_pickle) 
    ada_pickle.close()
elif alg == 'Soft Voting':
        # Load the pre-trained model from the pickle file
    sv_pickle = open('sv_fetal.pickle', 'rb') 
    clf = pickle.load(sv_pickle) 
    sv_pickle.close()

if file is not None:
    st.success('CSV file uploaded successfuly.', icon='✅')
    st.subheader(f'Predicting Fetal Health Class Using {alg} model')
    user_file = pd.read_csv(file)

    # used chat gpt here a bit to troubleshoot extracting the pred probability

    predictions = clf.predict(user_file)
    
    predictions = predictions.astype(int)
    pred_prob = clf.predict_proba(user_file)

    prob_predicted_class = pred_prob[np.arange(len(predictions)), predictions - 1]

    user_file['Predicted Fetal Health'] = predictions
    user_file['Prediction Probability'] = prob_predicted_class
    user_file['Prediction Probability'] = (user_file['Prediction Probability'] * 100).round(2)

    # used chat gpt to help color the cells

    user_file['Predicted Fetal Health'] = user_file['Predicted Fetal Health'].apply(lambda x: 'Normal' if x == 1 else ('Suspect' if x == 2 else 'Pathological'))

    # Apply custom color coding to the 'Fetal Health Prediction' column
    def color_coding(val):
        if val == 'Normal':
            return 'background-color: green'  # Normal
        elif val == 'Suspect':
            return 'background-color: yellow'  # Suspect
        elif val == 'Pathological':
            return 'background-color: orange'  # Pathological
        return ''  # No color if not matched

    # Apply the custom color coding
    styled_df = user_file.style.applymap(color_coding, subset=['Predicted Fetal Health'])

    # Display the styled DataFrame
    st.dataframe(styled_df)
    st.subheader('Model Performance and Insights')

else:
    st.info('*Please upload data to proceed*', icon='ℹ️')



#----------------------------------------------------------
if file is not None:
    if alg == 'Random Forest':
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('cmat_rf.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_rf.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 2: Feature Importance Visualization
        with tab4:
            st.write("### Feature Importance")
            st.image('feature_imp_rf.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

    if alg == 'Decision Tree':
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('cmat_dt.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_dt.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 2: Feature Importance Visualization
        with tab4:
            st.write("### Feature Importance")
            st.image('feature_imp_dt.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

    if alg == 'Ada Boost':
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('cmat_ada.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_ada.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 2: Feature Importance Visualization
        with tab4:
            st.write("### Feature Importance")
            st.image('feature_imp_ada.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")


    if alg == 'Soft Voting':
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('cmat_soft.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_soft.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 2: Feature Importance Visualization
        with tab4:
            st.write("### Feature Importance")
            st.image('feature_imp_s.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

