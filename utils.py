"""
Shared utility functions for the Osteosarcopenia Calculator web app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PKL_PATHS = {
    'osteoporosis': os.path.join(BASE_DIR, 'models', 'osteoporosis_screening_model.pkl'),
    'osteopenia': os.path.join(BASE_DIR, 'models', 'Osteopenia_screening_model.pkl'),
    'sarcopenia': os.path.join(BASE_DIR, 'models', 'Sarcopenia_screening_model.pkl'),
    'osteosarcopenia': os.path.join(BASE_DIR, 'models', 'osteosarcopenia_combined_model.pkl'),
    'osteosarcopenias': os.path.join(BASE_DIR, 'models', 'osteosarcopenias_combined_model.pkl'),
}

EXCEL_PATH = os.path.join(BASE_DIR, 'data', 'Osteosarcopenia_Comprehensive_Results.xlsx')
# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_pkl(path):
    """Load a pickle file."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file not found: {path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_excel_sheet(sheet_name):
    """Load a specific sheet from the Excel results file."""
    try:
        return pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error loading Excel sheet '{sheet_name}': {e}")
        return pd.DataFrame()

def get_threshold_for_model(all_results, model_name, default=0.5):
    """Get threshold for a specific model from all_results DataFrame."""
    if all_results is None or not isinstance(all_results, pd.DataFrame):
        return default
    
    # Check if Model is the index
    if 'Model' not in all_results.columns:
        if all_results.index.name == 'Model' or model_name in all_results.index:
            if model_name in all_results.index:
                return all_results.loc[model_name].get('Threshold', default)
        return default
    else:
        match = all_results[all_results['Model'] == model_name]
        if not match.empty:
            return match.iloc[0].get('Threshold', default)
    return default

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_metrics_table(df, key_prefix="model"):
    """Display a styled metrics table with model selection."""
    if df.empty:
        st.warning("No metrics data available.")
        return None
    
    # Make a copy and ensure 'Model' is a column (not index)
    df_display = df.copy()
    if 'Model' not in df_display.columns and df_display.index.name == 'Model':
        df_display = df_display.reset_index()
    elif 'Model' not in df_display.columns:
        # If index has no name but looks like model names, use it
        df_display['Model'] = df_display.index
        df_display = df_display.reset_index(drop=True)
    
    # Select columns to display
    display_cols = ['Model', 'Threshold', 'Sensitivity', 'Specificity', 'AUC', 'F2']
    available_cols = [c for c in display_cols if c in df_display.columns]
    
    # Format the dataframe for display
    display_df = df_display[available_cols].copy()
    for col in ['Threshold', 'Sensitivity', 'Specificity', 'AUC', 'F2']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Model selection
    model_names = df_display['Model'].tolist()
    selected_model = st.selectbox(
        "Select a model for prediction:",
        model_names,
        key=f"{key_prefix}_select"
    )
    
    return selected_model

def display_metrics_table_only(df):
    """Display a styled metrics table WITHOUT model selection dropdown."""
    if df.empty:
        st.warning("No metrics data available.")
        return
    
    # Make a copy and ensure 'Model' is a column (not index)
    df_display = df.copy()
    if 'Model' not in df_display.columns and df_display.index.name == 'Model':
        df_display = df_display.reset_index()
    elif 'Model' not in df_display.columns:
        df_display['Model'] = df_display.index
        df_display = df_display.reset_index(drop=True)
    
    # Select columns to display
    display_cols = ['Model', 'Threshold', 'Sensitivity', 'Specificity', 'AUC', 'F2']
    available_cols = [c for c in display_cols if c in df_display.columns]
    
    # Format the dataframe for display
    display_df = df_display[available_cols].copy()
    for col in ['Threshold', 'Sensitivity', 'Specificity', 'AUC', 'F2']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_combined_metrics_table(df, key_prefix="combined"):
    """Display metrics for combined models (Bone + Sarc) with selection dropdown."""
    if df.empty:
        st.warning("No combined model data available.")
        return None, None
    
    # Get top 15 by F2
    top15 = df.nlargest(15, 'F2').reset_index(drop=True)
    
    display_cols = ['Bone_Model', 'Sarc_Model', 'Threshold', 'Sensitivity', 'Specificity', 'F2']
    available_cols = [c for c in display_cols if c in top15.columns]
    
    display_df = top15[available_cols].copy()
    for col in ['Threshold', 'Sensitivity', 'Specificity', 'F2']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Create combined name for selection
    top15['Combined_Name'] = top15['Bone_Model'] + ' + ' + top15['Sarc_Model']
    selected = st.selectbox(
        "Select a combined model:",
        top15['Combined_Name'].tolist(),
        key=f"{key_prefix}_select"
    )
    
    if selected:
        row = top15[top15['Combined_Name'] == selected].iloc[0]
        return row['Bone_Model'], row['Sarc_Model']
    return None, None

def display_combined_metrics_table_only(df):
    """Display metrics for combined models WITHOUT selection dropdown."""
    if df.empty:
        st.warning("No combined model data available.")
        return
    
    # Get top 15 by F2
    top15 = df.nlargest(15, 'F2').reset_index(drop=True)
    
    display_cols = ['Bone_Model', 'Sarc_Model', 'Threshold', 'Sensitivity', 'Specificity', 'F2']
    available_cols = [c for c in display_cols if c in top15.columns]
    
    display_df = top15[available_cols].copy()
    for col in ['Threshold', 'Sensitivity', 'Specificity', 'F2']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================================
# INPUT FORM GENERATION
# ============================================================================

# Clinical labels for features (code -> display name)
FEATURE_LABELS = {
    'Sex': 'Sex',
    'Age': 'Age (years)',
    'Height': 'Height (cm)',
    'Weight': 'Weight (kg)',
    'Waistcm': 'Waist Circumference (cm)',
    'FemurRt': 'Thigh Circumference (cm)',
    'Forearm': 'Forearm Circumference (cm)',
    'HiPR03': 'Hormone Replacement Therapy',
    'HiPR06': 'Parity (Number of Live Births)',
    'LSSM01': 'Smoking Status',
    'Diabetes01': 'Diabetes',
    'CRF01': 'Chronic Renal Failure',
    'OSMU07': 'Previous Fracture Cause',
    'OSMU13': 'Fear of Falling',
    'OSMU16': 'Difficulty Lifting (~4.5 kg)',
    'OSMU17': 'Difficulty Walking Across Room',
    'OSMU19': 'Difficulty Climbing 10 Stairs',
    'MNA_Score': 'MNA Score (Nutritional Status)',
    'Osteoarth01': 'Osteoarthritis',
    'ADLDRESS': 'Independence in Dressing',
}

def generate_input_form(features, key_prefix="input"):
    """Generate input form for given features with clinical labels."""
    
    # Categorical features with labeled options
    # Format: { code: [(value, label), ...] }
    CATEGORICAL_OPTIONS = {
        'Sex': [('Female', 'Female'), ('Male', 'Male')],
        'Diabetes01': [(1.0, 'Yes'), (2.0, 'No')],
        'CRF01': [(1.0, 'Yes'), (2.0, 'No')],
        'HiPR03': [(0.0, 'Male / Not Applicable'), (1.0, 'Yes'), (2.0, 'No')],
        'LSSM01': [(1.0, 'Current Smoker'), (2.0, 'Former Smoker'), (3.0, 'Never Smoked')],
        'OSMU07': [(1.0, 'Severe Trauma / No History of Previous Fracture'), (2.0, 'Minimal Trauma (Fragility)')],
        'OSMU13': [(1.0, 'Yes'), (2.0, 'No')],
        'OSMU16': [(1.0, 'No Difficulty'), (2.0, 'Some Difficulty'), (3.0, 'Much Difficulty')],
        'OSMU17': [(1.0, 'No Difficulty'), (2.0, 'Some Difficulty'), (3.0, 'Much Difficulty')],
        'OSMU19': [(1.0, 'No Difficulty'), (2.0, 'Some Difficulty'), (3.0, 'Much Difficulty')],
        'Osteoarth01': [(1.0, 'Yes'), (2.0, 'No')],
        'ADLDRESS': [(1.0, 'Independent'), (2.0, 'Needs Some Help'), (3.0, 'Dependent')],
    }
    
    # Numerical ranges: (min, max, default)
    NUMERICAL_RANGES = {
        'Age': (40.0, 100.0, 65.0),
        'Height': (130.0, 190.0, 160.0),
        'Weight': (30.0, 120.0, 60.0),
        'Waistcm': (50.0, 150.0, 85.0),
        'FemurRt': (20.0, 100.0, 45.0),
        'Forearm': (15.0, 60.0, 25.0),
        'HiPR06': (0.0, 20.0, 0.0),
        'MNA_Score': (0.0, 14.0, 12.0),
    }
    
    cols = st.columns(3)
    input_data = {}
    
    for i, feat in enumerate(features):
        col = cols[i % 3]
        label = FEATURE_LABELS.get(feat, feat)
        
        if feat in CATEGORICAL_OPTIONS:
            options = CATEGORICAL_OPTIONS[feat]
            # Create format_func to show labels but return values
            option_values = [opt[0] for opt in options]
            option_labels = {opt[0]: opt[1] for opt in options}
            
            val = col.selectbox(
                label, 
                options=option_values, 
                format_func=lambda x, labels=option_labels: labels.get(x, str(x)),
                key=f"{key_prefix}_{feat}"
            )
        elif feat in NUMERICAL_RANGES:
            min_v, max_v, default = NUMERICAL_RANGES[feat]
            val = col.number_input(label, min_value=min_v, max_value=max_v, value=default, step=0.1, key=f"{key_prefix}_{feat}")
        else:
            # Default to number input
            val = col.number_input(label, min_value=0.0, step=1.0, key=f"{key_prefix}_{feat}")
        
        input_data[feat] = val
    
    return input_data
    
    return input_data

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_with_pkl(pkl_data, input_data, selected_model_name=None):
    """
    Make a prediction using the PKL data.
    
    If selected_model_name is None, uses the 'model' key (best model).
    Otherwise, uses the model from 'all_results' that matches the name.
    """
    if pkl_data is None:
        return None, None
    
    preprocessor = pkl_data.get('preprocessor')
    selected_features = pkl_data.get('selected_features', [])
    original_features = pkl_data.get('original_features', [])
    threshold = pkl_data.get('optimal_threshold', 0.5)
    
    # Get the model
    model = pkl_data.get('model')
    
    # If a specific model name is selected and we have trained_models dict
    trained_models = pkl_data.get('trained_models', {})
    all_results = pkl_data.get('all_results')
    
    if selected_model_name and trained_models and selected_model_name in trained_models:
        # Use the selected model from trained_models
        model_tuple = trained_models[selected_model_name]
        if isinstance(model_tuple, tuple):
            model, threshold = model_tuple
        else:
            model = model_tuple
            # Get threshold from all_results if available
            if all_results is not None and isinstance(all_results, pd.DataFrame):
                if 'Model' in all_results.columns:
                    match = all_results[all_results['Model'] == selected_model_name]
                elif all_results.index.name == 'Model' or selected_model_name in all_results.index:
                    match = all_results.loc[[selected_model_name]] if selected_model_name in all_results.index else pd.DataFrame()
                else:
                    match = pd.DataFrame()
                if not match.empty:
                    threshold = match.iloc[0].get('Threshold', threshold)
    elif all_results is not None and selected_model_name:
        # Fallback: Just get threshold from all_results (uses best model for prediction)
        if isinstance(all_results, pd.DataFrame):
            if 'Model' in all_results.columns:
                match = all_results[all_results['Model'] == selected_model_name]
            elif all_results.index.name == 'Model' or selected_model_name in all_results.index:
                match = all_results.loc[[selected_model_name]] if selected_model_name in all_results.index else pd.DataFrame()
            else:
                match = pd.DataFrame()
            if not match.empty:
                threshold = match.iloc[0].get('Threshold', threshold)
    
    # Create input DataFrame
    df = pd.DataFrame([input_data])
    
    try:
        # Check if preprocessor expects more columns
        if hasattr(preprocessor, 'feature_names_in_'):
            required_cols = preprocessor.feature_names_in_
            for c in required_cols:
                if c not in df.columns:
                    df[c] = np.nan  # Imputer will handle
        
        # Transform
        X_processed = preprocessor.transform(df)
        
        # Get feature names and filter to selected
        try:
            proc_names = preprocessor.get_feature_names_out()
            X_df = pd.DataFrame(X_processed, columns=proc_names)
            X_final = X_df[selected_features].values
        except Exception:
            X_final = X_processed
        
        # Predict
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_final)[:, 1][0]
        else:
            prob = float(model.predict(X_final)[0])
        
        is_positive = prob >= threshold
        
        return prob, is_positive, threshold
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, threshold

# ============================================================================
# STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling - Alice Blue professional medical theme."""
    st.markdown("""
    <style>
        /* Import professional medical font - Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Apply font globally */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main background - Alice Blue gradient */
        .stApp {
            background: linear-gradient(135deg, #F0F8FF 0%, #E6F2FF 50%, #F0F8FF 100%);
        }
        .main {
            background-color: transparent;
        }
        
        /* Text colors - professional dark */
        h1, h2, h3, h4, h5, h6 {
            color: #1e3a5f !important;
            font-weight: 700 !important;
        }
        h1 {
            font-weight: 800 !important;
        }
        p, span, label, .stMarkdown {
            color: #2c3e50 !important;
        }
        .stSelectbox label, .stNumberInput label {
            color: #1e3a5f !important;
            font-weight: 600 !important;
        }
        
        /* Dropdowns - white background with visible text (iOS fix) */
        .stSelectbox > div > div,
        .stSelectbox > div > div > div,
        [data-baseweb="select"],
        [data-baseweb="select"] > div,
        [data-baseweb="popover"] > div,
        div[data-baseweb="select"] > div {
            background-color: white !important;
            background: white !important;
            color: #1e3a5f !important;
            -webkit-appearance: none;
        }
        .stSelectbox > div > div {
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
        }
        /* Dropdown text visibility fix for iOS */
        .stSelectbox [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] div[aria-selected],
        .stSelectbox div[data-baseweb="select"] > div > div {
            color: #1e3a5f !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #1e3a5f !important;
        }
        
        /* Number inputs - white background with visible text */
        .stNumberInput > div > div > input,
        .stNumberInput input {
            background-color: white !important;
            background: white !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            color: #1e3a5f !important;
            -webkit-text-fill-color: #1e3a5f !important;
            opacity: 1 !important;
        }
        
        /* Result cards */
        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(30, 58, 95, 0.15);
            font-family: 'Inter', sans-serif;
        }
        .positive {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white !important;
        }
        .positive h1, .positive h2, .positive p {
            color: white !important;
        }
        .negative {
            background: linear-gradient(135deg, #27ae60 0%, #1e8449 100%);
            color: white !important;
        }
        .negative h1, .negative h2, .negative p {
            color: white !important;
        }
        
        /* Metric boxes */
        .metric-box {
            background: rgba(255,255,255,0.9);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem;
            border: 1px solid rgba(30, 58, 95, 0.1);
            box-shadow: 0 2px 8px rgba(30, 58, 95, 0.08);
        }
        
        /* Tables */
        .stDataFrame {
            background-color: white;
            border-radius: 8px;
        }
        
        /* Sidebar - professional blue */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a5f 0%, #2c5282 100%);
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox label {
            color: white !important;
        }
        
        /* Buttons - medical blue with bold visible text */
        .stButton>button {
            background: linear-gradient(135deg, #1e5a8a 0%, #2980b9 100%);
            color: white !important;
            border: none;
            font-weight: 800 !important;
            font-size: 1rem !important;
            font-family: 'Inter', sans-serif;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            letter-spacing: 0.5px;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #145374 0%, #1e5a8a 100%);
            box-shadow: 0 4px 15px rgba(41, 128, 185, 0.4);
        }
        .stButton>button p {
            color: white !important;
            font-weight: 800 !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255,255,255,0.7);
            border-radius: 8px 8px 0 0;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: white;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 500 !important;
            color: #1e3a5f !important;
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

def display_result(condition_name, probability, is_positive, threshold):
    """Display prediction result with styling."""
    status = "POSITIVE" if is_positive else "NEGATIVE"
    css_class = "positive" if is_positive else "negative"
    
    st.markdown(f"""
    <div class="result-card {css_class}">
        <h2>{condition_name}</h2>
        <h1>{status}</h1>
        <p style="font-size: 1.5rem;">Probability: {probability:.1%}</p>
        <p>Threshold: {threshold:.3f}</p>
    </div>
    """, unsafe_allow_html=True)
