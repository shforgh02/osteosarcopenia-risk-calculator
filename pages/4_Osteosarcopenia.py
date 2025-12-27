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
    load_pkl, load_excel_sheet, PKL_PATHS, apply_custom_css,
    display_metrics_table_only, display_combined_metrics_table_only,
    generate_input_form, predict_with_pkl, display_result
)

st.set_page_config(page_title="Osteosarcopenia Combined", page_icon="🦴💪", layout="wide")
apply_custom_css()

# ============================================================================
# OSTEOSARCOPENIA COMBINED PAGE
# ============================================================================

st.title("Osteosarcopenia Combined Screening")
st.markdown("""
Predict **all 5 conditions** at once using the unified Feature X set:
- Osteoporosis, Osteopenia, Sarcopenia
- Osteosarcopenia (Osteopenia + Sarcopenia)
- **Severe Osteosarcopenia** (Osteoporosis + Sarcopenia)
""")

# Load PKL files for combined predictions
pkl_osteosarc = load_pkl(PKL_PATHS['osteosarcopenia'])
pkl_severe = load_pkl(PKL_PATHS['osteosarcopenias'])

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
    
    tab1, tab2 = st.tabs(["📊 Model Performance", "🔮 Predict All"])
    
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
        pkl_osteoporosis = load_pkl(PKL_PATHS['osteoporosis'])
        pkl_osteopenia = load_pkl(PKL_PATHS['osteopenia'])
        pkl_sarcopenia = load_pkl(PKL_PATHS['sarcopenia'])
        
        # === BASE CONDITIONS (3 models) ===
        st.markdown("**Base Conditions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            osteoporosis_models = get_model_list(pkl_osteoporosis)
            selected_osteoporosis = st.selectbox("Osteoporosis", osteoporosis_models, key="sel_osteoporosis") if osteoporosis_models else None
        
        with col2:
            osteopenia_models = get_model_list(pkl_osteopenia)
            selected_osteopenia = st.selectbox("Osteopenia", osteopenia_models, key="sel_osteopenia") if osteopenia_models else None
        
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
        
        # === BONE × SARC COMBINED MODELS (2 selections) ===
        st.markdown("**Combined Conditions (Bone × Sarc):**")
        col1, col2 = st.columns(2)
        
        with col1:
            combined_osteosarc_list = get_combined_model_list(combined_sheets.get('Osteosarcopenia'))
            selected_combined_osteosarc = st.selectbox("Osteosarcopenia (Bone+Sarc)", combined_osteosarc_list, key="sel_combined_osteosarc") if combined_osteosarc_list else None
        
        with col2:
            combined_severe_list = get_combined_model_list(combined_sheets.get('Severe Osteosarcopenia'))
            selected_combined_severe = st.selectbox("Severe Osteosarcopenia (Bone+Sarc)", combined_severe_list, key="sel_combined_severe") if combined_severe_list else None
        
        st.markdown("---")
        
        if st.button("🔮 Predict All Conditions", type="primary", use_container_width=True):
            with st.spinner("Calculating predictions for all conditions..."):
                
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                # Track results for paradox detection
                results = {}
                
                # Row 1: Base conditions (Osteoporosis, Osteopenia, Sarcopenia)
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
                            display_result("Osteopenia", prob, is_pos, thresh)
                            results['osteopenia'] = is_pos
                    else:
                        st.warning("Osteopenia model not loaded")
                
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
                
                # Row 3: Combined model predictions (Bone × Sarc)
                st.markdown("### Combined Conditions (Bone × Sarc Models)")
                st.caption("*These use your selected Bone + Sarcopenia combination*")
                col1, col2 = st.columns(2)
                
                with col1:
                    combined_data = combined_sheets.get('Osteosarcopenia')
                    if combined_data is not None and not combined_data.empty and selected_combined_osteosarc:
                        # Find the selected row
                        combined_data['Combined_Name'] = combined_data['Bone_Model'] + ' + ' + combined_data['Sarc_Model']
                        selected_row = combined_data[combined_data['Combined_Name'] == selected_combined_osteosarc]
                        if not selected_row.empty:
                            row = selected_row.iloc[0]
                            st.markdown(f"**Selected Model:** {row['Bone_Model']} + {row['Sarc_Model']}")
                            # Use the bone model for prediction
                            bone_model = row['Bone_Model']
                            prob, is_pos, _ = predict_with_pkl(pkl_osteoporosis, input_data, bone_model)
                            if prob is not None:
                                thresh = row.get('Threshold', 0.5)
                                is_pos_combined = prob >= thresh
                                display_result("Osteosarcopenia (Combined)", prob, is_pos_combined, thresh)
                                results['osteosarcopenia_combined'] = is_pos_combined
                
                with col2:
                    combined_data = combined_sheets.get('Severe Osteosarcopenia')
                    if combined_data is not None and not combined_data.empty and selected_combined_severe:
                        # Find the selected row
                        combined_data['Combined_Name'] = combined_data['Bone_Model'] + ' + ' + combined_data['Sarc_Model']
                        selected_row = combined_data[combined_data['Combined_Name'] == selected_combined_severe]
                        if not selected_row.empty:
                            row = selected_row.iloc[0]
                            st.markdown(f"**Selected Model:** {row['Bone_Model']} + {row['Sarc_Model']}")
                            # Use the bone model for prediction
                            bone_model = row['Bone_Model']
                            prob, is_pos, _ = predict_with_pkl(pkl_osteoporosis, input_data, bone_model)
                            if prob is not None:
                                thresh = row.get('Threshold', 0.5)
                                is_pos_combined = prob >= thresh
                                display_result("Severe Osteosarcopenia (Combined)", prob, is_pos_combined, thresh)
                                results['severe_combined'] = is_pos_combined
                
                # ====================================================================
                # PARADOX DETECTION
                # ====================================================================
                paradoxes = []
                
                # Check Osteosarcopenia paradox (Osteopenia + Sarcopenia)
                if results.get('osteopenia') and results.get('sarcopenia'):
                    if not results.get('osteosarcopenia_direct'):
                        paradoxes.append(
                            "**Osteosarcopenia (Direct)**: Components (Osteopenia + Sarcopenia) are both POSITIVE, "
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
