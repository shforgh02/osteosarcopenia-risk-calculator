"""
Page 3: Sarcopenia Prediction
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_pkl, PKL_PATHS, apply_custom_css,
    display_metrics_table, generate_input_form,
    predict_with_pkl, display_result, get_threshold_for_model
)

st.set_page_config(page_title="Sarcopenia - Risk Calculator", page_icon="💪", layout="wide")
apply_custom_css()

# ============================================================================
# SARCOPENIA PAGE
# ============================================================================

st.title("Sarcopenia Screening")
st.markdown("Predict the risk of **Sarcopenia** (low muscle mass and function) using machine learning.")

pkl_data = load_pkl(PKL_PATHS['sarcopenia'])

if pkl_data:
    original_features = pkl_data.get('original_features', [])
    all_results = pkl_data.get('all_results')
    
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    st.markdown("*Select which of the 15 models you want to use for prediction:*")
    
    if all_results is not None and isinstance(all_results, pd.DataFrame):
        selected_model = display_metrics_table(all_results, key_prefix="sarcopenia")
        threshold = get_threshold_for_model(all_results, selected_model, pkl_data.get('optimal_threshold', 0.5))
    else:
        st.info("Using the best model.")
        selected_model = None
        threshold = pkl_data.get('optimal_threshold', 0.5)
    
    st.markdown("---")
    st.subheader("📝 Enter Patient Data")
    st.markdown(f"*Features required: {len(original_features)}*")
    
    input_data = generate_input_form(original_features, key_prefix="sarc_input")
    
    if st.button("Predict Sarcopenia Risk", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            prob, is_positive, threshold = predict_with_pkl(pkl_data, input_data, selected_model)
            
            if prob is not None:
                st.markdown("---")
                st.subheader("📋 Prediction Result")
                display_result("Sarcopenia", prob, is_positive, threshold)
            else:
                st.error("Prediction failed.")

else:
    st.error("Could not load Sarcopenia model.")
