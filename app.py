import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

@st.cache_data
def train_model():
    data = pd.read_csv('winequality-red.csv')
    X = data.drop('quality', axis=1)
    y = data['quality']

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    model = SVR(kernel='rbf')
    model.fit(X_scaled, y_scaled)

    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = train_model()


st.set_page_config(page_title="Wine Quality Prediction", layout="wide")

st.markdown("""
<style>
body { 
font-family: 'Raleway',
sans-serif;
background-color: #f8f3ed; 
font-size: 15px;
}
h1, h2 { 
font-family: 'Playfair Display', 
serif;
}
.header-font { 
font-family: 'Playfair Display',
serif; }
.wine-red {
color: #722f37;
}
.bg-wine { 
background-color: black; 
color: white;
padding: 20px;
border-radius: 8px;
}
.form-box {
background-color: #722f37;
padding: 20px;
border-radius: 8px; 
box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.result-box { 
background-color: #722f37; 
padding: 20px; 
border-radius: 8px; 
box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
}
.progress {
height: 16px; 
background-color: #e2e8f0; 
border-radius: 8px; 
overflow: hidden;
}
.progress-bar {
height: 100%; 
background-color: #722f37;
width: 0%;
transition: width 0.5s; 
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="bg-wine"><h1 class="header-font" style="text-align:center;">Dự Đoán Chất Lượng Rượu Vang</h1></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="form-box"><h2 class="wine-red header-font">Input Wine Parameters</h2>', unsafe_allow_html=True)
    fixed_acidity = st.number_input("Fixed Acidity")
    volatile_acidity = st.number_input("Volatile Acidity")
    citric_acid = st.number_input("Citric Acid")
    residual_sugar = st.number_input("Residual Sugar")
    chlorides = st.number_input("Chlorides")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
    density = st.number_input("Density")
    pH = st.number_input("pH")
    sulphates = st.number_input("Sulphates")
    alcohol = st.number_input("Alcohol")
    predict_btn = st.button("Predict Wine Quality")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-box"><h2 class="wine-red header-font">Prediction Results</h2>', unsafe_allow_html=True)
    if predict_btn:
        try:
            features = np.array([[
                fixed_acidity, volatile_acidity, citric_acid,
                residual_sugar, chlorides, free_sulfur_dioxide,
                total_sulfur_dioxide, density, pH, sulphates, alcohol
            ]])
            features_scaled = scaler_X.transform(features)
            prediction_scaled = model.predict(features_scaled)[0]
            prediction = scaler_y.inverse_transform([[prediction_scaled]])[0][0]
            prediction = round(prediction, 2)

            st.markdown(f"""
                <div class="wine-red" style="font-size: 24px; font-weight: bold;">
                    Quality Score (0-10): {prediction}
                </div>
                <div class="progress">
                    <div class="progress-bar" style="width: {min(100, max(0, prediction/10*100))}%"></div>
                </div>
                """, unsafe_allow_html=True)

            if prediction >= 7:
                st.success("✅ Excellent quality wine!")
                st.image("https://cdn-icons-png.flaticon.com/512/979/979585.png", width=100)
            elif prediction >=5:
                st.warning("⚠️ Good quality wine.")
                st.image("https://cdn-icons-png.flaticon.com/512/5793/5793147.png", width=100)
            else:
                st.error("🚫 Low quality wine.")
                st.image("https://cdn-icons-png.flaticon.com/512/7556/7556216.png", width=100)
        except Exception as e:
            st.error(f"🚨 Error: {e}")
    else:
        st.info("Nhập thông tin rượu và bấm Predict để xem kết quả.")
    st.markdown('</div>', unsafe_allow_html=True)
