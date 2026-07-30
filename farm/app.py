"""
FarmIQ - Smart Agri Advisor Streamlit Dashboard
"""

import os
import subprocess
import streamlit as st

# ---------- FIX: Cloud पर Models खुद Train होंगे (Storage बचाने के लिए) ----------
if not os.path.exists('models/crop_yield_model.pkl'):
    st.warning("⚙️ Models not found! Training AI models on Cloud... (Takes ~20 seconds)")
    subprocess.run(['python', 'train_model.py'], check=True)
    st.success("✅ Models trained! Refreshing...")
    st.rerun()
# ---------- बाकी का आपका Original Code यहाँ से शुरू करें ----------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

st.set_page_config(
    page_title="FarmIQ - Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Header
try:
    st.image("https://img.freepik.com/premium-vector/modern-farm-logo-vector_658271-1527.jpg?w=360", width=80)
except:
    st.title("🌾 FarmIQ")
st.title("🌾 Smart Agri Advisor - Crop Yield Prediction")
st.markdown("---")

# ---------- Load Models ----------
base_dir = os.path.dirname(os.path.abspath(__file__))

def load_file(filename):
    """Load a pickle/joblib file from various possible locations."""
    paths = [
        os.path.join(base_dir, 'models', filename),
        os.path.join(base_dir, 'src', 'models', filename),
        os.path.join(base_dir, filename),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except:
                import pickle
                with open(p, 'rb') as f:
                    return pickle.load(f)
    raise FileNotFoundError(f"{filename} not found in any of {paths}")

try:
    model = load_file('crop_yield_model.pkl')
    label_encoder = load_file('label_encoder.pkl')
    scaler = load_file('scaler.pkl')
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.info("Please run 'python train_model.py' first to train and save the models.")
    st.stop()

# ---------- Input Section ----------
st.subheader("📊 Enter Field Parameters")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("🌱 Nitrogen (N)", 0.0, 200.0, 50.0, step=1.0)
    phosphorus = st.number_input("🌱 Phosphorus (P)", 0.0, 200.0, 50.0, step=1.0)
    potassium = st.number_input("🌱 Potassium (K)", 0.0, 200.0, 50.0, step=1.0)
    temperature = st.number_input("🌡️ Temperature (°C)", -10.0, 50.0, 25.0, step=0.5)

with col2:
    humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 60.0, step=1.0)
    ph = st.number_input("🧪 pH Level", 0.0, 14.0, 7.0, step=0.1)
    rainfall = st.number_input("☔ Rainfall (mm)", 0.0, 500.0, 100.0, step=5.0)
    
    # Crop selection
    if hasattr(label_encoder, 'classes_'):
        crop_options = list(label_encoder.classes_)
    else:
        crop_options = ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", 
                        "Groundnut", "Soybean", "Potato", "Onion", "Tomato"]
    crop = st.selectbox("🌾 Crop Type", crop_options)

# ---------- Prediction ----------
if st.button("🚀 Predict Yield", type="primary"):
    try:
        # Encode crop
        crop_enc = label_encoder.transform([crop])[0]
        
        # Prepare features
        features = np.array([[nitrogen, phosphorus, potassium, temperature, 
                              humidity, ph, rainfall, crop_enc]])
        
        # Scale features
        try:
            features = scaler.transform(features)
        except:
            pass  # If scaler fails, use raw features
        
        # Predict
        pred = model.predict(features)[0]
        
        # Display result
        st.success(f"🌾 Predicted Yield: **{pred:.2f} tons/ha**")
        
        # ---------- Profit Analysis ----------
        st.subheader("💰 Profit Analysis")
        land_area = st.number_input("Land Area (hectares)", 1.0, 100.0, 1.0, step=0.5)
        
        from src.data_loader import ProfitCalculator
        calc = ProfitCalculator(land_area)
        profit_data = calc.calculate_profit(crop, pred)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Yield", f"{profit_data['total_yield_tons']:.2f} tons")
        col_b.metric("Revenue", f"₹{profit_data['revenue']:,.2f}")
        col_c.metric("Total Cost", f"₹{profit_data['total_cost']:,.2f}")
        col_d.metric("Net Profit", f"₹{profit_data['profit']:,.2f}")
        
        st.metric("ROI", f"{profit_data['roi_percentage']:.1f}%")
        
        # ---------- Fertilizer Recommendations ----------
        st.subheader("🧪 Fertilizer Recommendations")
        recs = ProfitCalculator.get_fertilizer_recommendation(nitrogen, phosphorus, potassium, ph)
        for rec in recs:
            st.write(f"- {rec}")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.text(traceback.format_exc())

st.markdown("---")
st.caption("🌾 FarmIQ - Smart Agri Advisor | Built with ❤️ for farmers")
