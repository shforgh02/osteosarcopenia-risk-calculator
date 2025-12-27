"""
Page 1: Osteoporosis Prediction
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_pkl, PKL_PATHS, apply_custom_css,
    display_metrics_table, generate_input_form,
    predict_with_pkl, display_result, get_threshold_for_model
)

# Page config
st.set_page_config(page_title="Osteoporosis - Risk Calculator", page_icon="🦴", layout="wide")
apply_custom_css()

# ============================================================================
# OSTEOPOROSIS PAGE
# ============================================================================

st.title("Osteoporosis Screening")
st.markdown("Predict the risk of **Osteoporosis** (severe bone density loss) using machine learning.")

# Load model data
pkl_data = load_pkl(PKL_PATHS['osteoporosis'])

if pkl_data:
    # Get features and results
    original_features = pkl_data.get('original_features', [])
    all_results = pkl_data.get('all_results')
    
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    st.markdown("*Select which of the 15 models you want to use for prediction:*")
    
    # Display metrics table
    if all_results is not None and isinstance(all_results, pd.DataFrame):
        selected_model = display_metrics_table(all_results, key_prefix="osteoporosis")
        threshold = get_threshold_for_model(all_results, selected_model, pkl_data.get('optimal_threshold', 0.5))
    else:
        st.info("Using the best model (no detailed results available).")
        selected_model = None
        threshold = pkl_data.get('optimal_threshold', 0.5)
    
    st.markdown("---")
    st.subheader("📝 Enter Patient Data")
    st.markdown(f"*Features required: {len(original_features)}*")
    
    # Generate input form
    input_data = generate_input_form(original_features, key_prefix="osteo_input")
    
    # Predict button
    if st.button("Predict Osteoporosis Risk", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            prob, is_positive, threshold = predict_with_pkl(pkl_data, input_data, selected_model)
            
            if prob is not None:
                st.markdown("---")
                st.subheader("📋 Prediction Result")
                display_result("Osteoporosis", prob, is_positive, threshold)
            else:
                st.error("Prediction failed. Please check inputs.")

else:
    st.error("Could not load Osteoporosis model. Please ensure the PKL file exists.")
