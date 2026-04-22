
import streamlit as st
import pickle
import numpy as np

# Set page title
st.set_page_config(page_title='Power Consumption Predictor', layout='centered')

# Load models
@st.cache_resource
def load_models():
    with open('models.pkl', 'rb') as f:
        return pickle.load(f)

data = load_models()

st.title('⚡ Household Power Prediction')
st.markdown('Deploy model menggunakan Streamlit & Pickle')

# Sidebar for model selection
st.sidebar.header('Configuration')
algo = st.sidebar.selectbox('Pilih Algoritma:', ['Lasso', 'Decision Tree', 'Polynomial Regression'])

# Input fields in the main area
st.subheader('Input Atribut Pengukuran')
col1, col2 = st.columns(2)

with col1:
    reactive = st.number_input('Global Reactive Power (kW)', value=0.12)
    voltage = st.number_input('Voltage (V)', value=240.0)
    intensity = st.number_input('Global Intensity (A)', value=11.0)
    hour = st.slider('Jam (0-23)', 0, 23, 12)

with col2:
    sub1 = st.number_input('Sub Metering 1 (Dapur)', value=1.2)
    sub2 = st.number_input('Sub Metering 2 (Cuci)', value=1.3)
    sub3 = st.number_input('Sub Metering 3 (Pemanas/AC)', value=6.5)

# Prediction Logic
if st.button('Predict Consumption'):
    # Prepare input array
    input_features = np.array([[reactive, voltage, intensity, sub1, sub2, sub3, hour]])
    
    if algo == 'Lasso':
        prediction = data['lasso'].predict(input_features)
    elif algo == 'Decision Tree':
        prediction = data['dt'].predict(input_features)
    else:
        poly_input = data['poly_features'].transform(input_features)
        prediction = data['poly_model'].predict(poly_input)

    st.success(f'Hasil Prediksi Global Active Power: {prediction[0]:.4f} kW')
