import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="DrivePrice", page_icon="🚗", layout="centered")
st.markdown("<div id='top'></div>", unsafe_allow_html=True)

# --- CUSTOM CSS (Modern UI) ---
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #f0f2f6; }
    
    /* Input Container Styling */
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Professional Result Card */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .price-text {
        font-size: 55px;
        font-weight: 800;
        color: #38bdf8;
        margin: 10px 0;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }

    .conf-label {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
model = pickle.load(open('car_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
cols = pickle.load(open('columns.pkl', 'rb'))

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #1e293b;'>🚗 DrivePrice</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Get the most accurate market valuation for your car</p>", unsafe_allow_html=True)

# --- INPUT SECTION ---
st.write("### 📝 Vehicle Details")
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Car Brand", ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra", "Ford", "BMW", "Mercedes-Benz", "Other"])
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=30000, step=500)

with col2:
    car_age = st.slider("Vehicle Age (Years)", 0, 20, 5)
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

owner = st.selectbox("Previous Owners", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

st.markdown("---")

# --- PREDICTION ---
if st.button("Generate Valuation Report", use_container_width=True):
    # Prepare Data
    input_df = pd.DataFrame(0, index=[0], columns=cols)
    input_df['km_driven'] = float(km_driven)
    input_df['car_age'] = float(car_age)
    
    if f'brand_{brand.lower()}' in cols: input_df[f'brand_{brand.lower()}'] = 1
    if f'owner_{owner.lower()}' in cols: input_df[f'owner_{owner.lower()}'] = 1
    if transmission == "Manual" and 'transmission_manual' in cols: input_df['transmission_manual'] = 1

    # Predict
    input_scaled = scaler.transform(input_df)
    log_price = model.predict(input_scaled)
    final_price = np.exp(log_price)[0] / 100000

    # DISPLAY PREMIUM RESULT
    st.markdown(f"""
        <div class="result-card">
            <p class="conf-label">Estimated Resale Value</p>
            <p class="price-text">₹ {final_price:.2f} Lakhs</p>
            <p style="color: #94a3b8; font-size: 14px;">Market valuation for a {brand} in current condition</p>
            <hr style="border-color: #334155; margin: 20px 0;">
            <div style="display: flex; justify-content: space-around; font-size: 13px;">
                <span>⚙️ {transmission}</span>
                <span>📅 {car_age} Years Old</span>
                <span>📍 {km_driven} KM</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---") # Ek patli line separator ke liye

# Expander use karne se UI clean rehta hai
with st.expander("🛠️ Technical Details & Model Performance"):
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(label="Model Accuracy (R² Score)", value="78.4%") # Apni exact accuracy likhein
        st.caption("Based on 80/20 Train-Test split validation.")
        
    with col_b:
        st.write("**Algorithm:** Linear Regression")
        st.write("**Features:** 26 One-Hot Encoded Variables")
        
    st.info("The model uses a Log-Linear approach to ensure prices remain positive and follow market depreciation curves.")    

    # Success message without intrusive balloons
    st.toast('Valuation Report Generated!', icon='✅')