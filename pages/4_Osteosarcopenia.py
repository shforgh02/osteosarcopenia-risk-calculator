"""
Page 4: Osteosarcopenia Combined Prediction
Predicts all 5 conditions at once using Feature X (unified feature set).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_pkl, load_excel_sheet, PKL_PATHS_FEATUREX, PKL_PATHS_STACKING, apply_custom_css,
    display_metrics_table_only, display_combined_metrics_table_only,
    generate_input_form, predict_with_pkl, predict_with_stacking, display_result
)

st.set_page_config(page_title="Osteosarcopenia Combined", page_icon="🦴💪", layout="wide")
apply_custom_css()

# ============================================================================
# OSTEOSARCOPENIA COMBINED PAGE
# ============================================================================

st.title("Osteosarcopenia Combined Screening")
st.markdown("""
Predict **all 5 conditions** at once using the unified Feature X set:
- Osteoporosis, Low Bone Mass, Sarcopenia
- Osteosarcopenia (Low Bone Mass + Sarcopenia)
- **Severe Osteosarcopenia** (Osteoporosis + Sarcopenia)
""")

# Load PKL files for combined predictions
pkl_osteosarc = load_pkl(PKL_PATHS_FEATUREX['osteosarcopenia'])
pkl_severe = load_pkl(PKL_PATHS_FEATUREX['osteosarcopenias'])

# Load Excel sheets for metrics
direct_sheets = {
    'Osteoporosis': load_excel_sheet('Direct_Osteoporosis'),
    'Osteopenia': load_excel_sheet('Direct_Osteopenia'),
    'Sarcopenia': load_excel_sheet('Direct_Sarcopenia'),
    'Osteosarcopenia': load_excel_sheet('Direct_Osteosarcopenia'),
    'Severe Osteosarcopenia': load_excel_sheet('Direct_Osteosarcopenias'),
}

combined_sheets = {
    'Osteosarcopenia': load_excel_sheet('Combined_Osteosarcopenia_X'),
    'Severe Osteosarcopenia': load_excel_sheet('Combined_Osteosarcopenias_X'),
}

if pkl_osteosarc:
    original_features = pkl_osteosarc.get('original_features', [])
    
    st.markdown("---")
    
    # ========================================================================
    # TABS FOR ORGANIZATION
    # ========================================================================
    
    tab1, tab2 = st.tabs(["📊 Model Performance", "Predict All"])
    
    # ========================================================================
    # TAB 1: MODEL METRICS
    # ========================================================================
    
    with tab1:
        st.subheader("Direct Model Performance (15 models each)")
        st.markdown("*These are single models trained directly on each target.*")
        
        for condition, df in direct_sheets.items():
            with st.expander(f"📈 {condition}", expanded=False):
                if not df.empty:
                    display_metrics_table_only(df)
                else:
                    st.warning(f"No data for {condition}")
        
        st.markdown("---")
        st.subheader("Combined Model Performance (Top 15 by F2)")
        st.markdown("*These combine a Bone model + Sarcopenia model for joint prediction.*")
        
        for condition, df in combined_sheets.items():
            with st.expander(f"🔗 {condition} (Bone + Sarc)", expanded=False):
                if not df.empty:
                    display_combined_metrics_table_only(df)
                else:
                    st.warning(f"No combined data for {condition}")
    
    # ========================================================================
    # TAB 2: PREDICTION
    # ========================================================================
    
    with tab2:
        st.subheader("📝 Enter Patient Data")
        st.markdown(f"*Using Feature X: {len(original_features)} features*")
        
        # Input form
        input_data = generate_input_form(original_features, key_prefix="combined_input")
        
        st.markdown("---")
        
        # Model Selection Section
        st.subheader("🔧 Select Models for Each Condition")
        
        # Get model lists from all_results
        def get_model_list(pkl_data):
            if pkl_data is None:
                return []
            all_results = pkl_data.get('all_results')
            if all_results is None or not isinstance(all_results, pd.DataFrame):
                return []
            if 'Model' in all_results.columns:
                return all_results['Model'].tolist()
            elif all_results.index.name == 'Model':
                return all_results.index.tolist()
            return all_results.index.tolist()
        
        def get_combined_model_list(df):
            """Get list of combined model names from Excel sheet."""
            if df is None or df.empty:
                return []
            top15 = df.nlargest(15, 'F2').reset_index(drop=True)
            return (top15['Bone_Model'] + ' + ' + top15['Sarc_Model']).tolist()
        
        # Load all PKLs to get their model lists
        pkl_osteoporosis = load_pkl(PKL_PATHS_FEATUREX['osteoporosis'])
        pkl_osteopenia = load_pkl(PKL_PATHS_FEATUREX['osteopenia'])
        pkl_sarcopenia = load_pkl(PKL_PATHS_FEATUREX['sarcopenia'])
        
        # === BASE CONDITIONS (3 models) ===
        st.markdown("**Base Conditions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            osteoporosis_models = get_model_list(pkl_osteoporosis)
            selected_osteoporosis = st.selectbox("Osteoporosis", osteoporosis_models, key="sel_osteoporosis") if osteoporosis_models else None
        
        with col2:
            osteopenia_models = get_model_list(pkl_osteopenia)
            selected_osteopenia = st.selectbox("Low Bone Mass", osteopenia_models, key="sel_osteopenia") if osteopenia_models else None
        
        with col3:
            sarcopenia_models = get_model_list(pkl_sarcopenia)
            selected_sarcopenia = st.selectbox("Sarcopenia", sarcopenia_models, key="sel_sarcopenia") if sarcopenia_models else None
        
        # === DIRECT COMBINED CONDITIONS (2 models) ===
        st.markdown("**Direct Combined Conditions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            osteosarc_models = get_model_list(pkl_osteosarc)
            selected_osteosarc = st.selectbox("Osteosarcopenia (Direct)", osteosarc_models, key="sel_osteosarc") if osteosarc_models else None
        
        with col2:
            severe_models = get_model_list(pkl_severe)
            selected_severe = st.selectbox("Severe Osteosarcopenia (Direct)", severe_models, key="sel_severe") if severe_models else None
        
        # === STACKING COMBINED MODELS (Meta-Learner, 2 selections) ===
        st.markdown("**Combined Conditions (Meta-Learner Stacking):**")
        col1, col2 = st.columns(2)
        
        # Load stacking PKLs to get model lists
        pkl_stacking_osteosarc = load_pkl(PKL_PATHS_STACKING['osteosarcopenia'])
        pkl_stacking_severe = load_pkl(PKL_PATHS_STACKING['osteosarcopenias'])
        
        with col1:
            stacking_osteosarc_models = pkl_stacking_osteosarc.get('model_names', []) if pkl_stacking_osteosarc else []
            selected_stacking_osteosarc = st.selectbox("Osteosarcopenia (Stacking)", stacking_osteosarc_models, key="sel_stacking_osteosarc") if stacking_osteosarc_models else None
        
        with col2:
            stacking_severe_models = pkl_stacking_severe.get('model_names', []) if pkl_stacking_severe else []
            selected_stacking_severe = st.selectbox("Severe Osteosarcopenia (Stacking)", stacking_severe_models, key="sel_stacking_severe") if stacking_severe_models else None
        
        st.markdown("---")
        
        if st.button("Predict All Conditions", type="primary", use_container_width=True):
            with st.spinner("Calculating predictions for all conditions..."):
                
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                # Track results for paradox detection
                results = {}
                
                # Row 1: Base conditions (Osteoporosis, Low Bone Mass, Sarcopenia)
                st.markdown("### Base Conditions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if pkl_osteoporosis:
                        prob, is_pos, thresh = predict_with_pkl(pkl_osteoporosis, input_data, selected_osteoporosis)
                        if prob is not None:
                            display_result("Osteoporosis", prob, is_pos, thresh)
                            results['osteoporosis'] = is_pos
                    else:
                        st.warning("Osteoporosis model not loaded")
                
                with col2:
                    if pkl_osteopenia:
                        prob, is_pos, thresh = predict_with_pkl(pkl_osteopenia, input_data, selected_osteopenia)
                        if prob is not None:
                            display_result("Low Bone Mass", prob, is_pos, thresh)
                            results['osteopenia'] = is_pos
                    else:
                        st.warning("Low Bone Mass model not loaded")
                
                with col3:
                    if pkl_sarcopenia:
                        prob, is_pos, thresh = predict_with_pkl(pkl_sarcopenia, input_data, selected_sarcopenia)
                        if prob is not None:
                            display_result("Sarcopenia", prob, is_pos, thresh)
                            results['sarcopenia'] = is_pos
                    else:
                        st.warning("Sarcopenia model not loaded")
                
                # Row 2: Direct combined conditions
                st.markdown("### Combined Conditions (Direct Models)")
                col1, col2 = st.columns(2)
                
                with col1:
                    prob, is_pos, thresh = predict_with_pkl(pkl_osteosarc, input_data, selected_osteosarc)
                    if prob is not None:
                        display_result("Osteosarcopenia", prob, is_pos, thresh)
                        results['osteosarcopenia_direct'] = is_pos
                
                with col2:
                    prob, is_pos, thresh = predict_with_pkl(pkl_severe, input_data, selected_severe)
                    if prob is not None:
                        display_result("Severe Osteosarcopenia", prob, is_pos, thresh)
                        results['severe_direct'] = is_pos
                
                # Row 3: Combined model predictions (Stacking - FlawlessCombinedModel)
                st.markdown("### Combined Conditions (Meta-Learner Stacking)")
                st.caption("*Uses selected model from dropdown above*")
                col1, col2 = st.columns(2)
                
                with col1:
                    if pkl_stacking_osteosarc and selected_stacking_osteosarc:
                        prob, is_pos, thresh = predict_with_stacking(pkl_stacking_osteosarc, input_data, selected_stacking_osteosarc)
                        if prob is not None:
                            display_result("Osteosarcopenia (Stacking)", prob, is_pos, thresh)
                            results['osteosarcopenia_stacking'] = is_pos
                    else:
                        st.warning("Stacking model not found - run training script first")
                
                with col2:
                    if pkl_stacking_severe and selected_stacking_severe:
                        prob, is_pos, thresh = predict_with_stacking(pkl_stacking_severe, input_data, selected_stacking_severe)
                        if prob is not None:
                            display_result("Severe Osteosarcopenia (Stacking)", prob, is_pos, thresh)
                            results['severe_stacking'] = is_pos
                    else:
                        st.warning("Stacking model not found - run training script first")
                
                # ====================================================================
                # PARADOX DETECTION
                # ====================================================================
                paradoxes = []
                
                # Check Osteosarcopenia paradox (Low Bone Mass + Sarcopenia)
                if results.get('osteopenia') and results.get('sarcopenia'):
                    if not results.get('osteosarcopenia_direct'):
                        paradoxes.append(
                            "**Osteosarcopenia (Direct)**: Components (Low Bone Mass + Sarcopenia) are both POSITIVE, "
                            "but the direct model predicts NEGATIVE. This may be due to the higher threshold (0.686) "
                            "used by the direct model vs. the combined model approach."
                        )
                
                # Check Severe Osteosarcopenia paradox (Osteoporosis + Sarcopenia)
                if results.get('osteoporosis') and results.get('sarcopenia'):
                    if not results.get('severe_direct'):
                        paradoxes.append(
                            "**Severe Osteosarcopenia (Direct)**: Components (Osteoporosis + Sarcopenia) are both POSITIVE, "
                            "but the direct model predicts NEGATIVE. This may be due to different threshold optimization."
                        )
                
                # Display warnings if paradoxes detected
                if paradoxes:
                    st.markdown("---")
                    st.warning("⚠️ **Model Disagreement Detected**")
                    st.markdown("""
                    The following logical inconsistencies were detected between component and combined predictions. 
                    This can occur due to:
                    - Different threshold optimization strategies (F2 vs constrained)
                    - Different model architectures and their decision boundaries
                    - Different resampling strategies during training
                    
                    **This is expected behavior in research settings and highlights the complexity of multi-condition prediction.**
                    """)
                    for p in paradoxes:
                        st.markdown(f"- {p}")

else:
    st.error("Could not load combined model files. Please ensure PKL files exist.")
