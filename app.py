"""
Osteosarcopenia Risk Calculator - Main Entry Point

A multi-page Streamlit application for predicting bone and muscle health conditions.
"""

import streamlit as st

# Page config must be first
st.set_page_config(
    page_title="Osteosarcopenia Risk Calculator",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import utils after page config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import apply_custom_css

# Apply styling
apply_custom_css()

# ============================================================================
# HOME PAGE
# ============================================================================

st.title("Osteosarcopenia Risk Calculator")

st.markdown("""
## Welcome!

This application predicts the risk of **bone and muscle health conditions** using advanced machine learning models.

### Available Predictions

| Page | Condition | Description |
|------|-----------|-------------|
| **1. Osteoporosis** | Low bone density (BMD T-score of ≤ -2.5) | Uses dedicated screening model |
| **2. Low Bone Mass** | Low bone density (BMD T-score ≤ −1.0) | Uses dedicated screening model |
| **3. Sarcopenia** | Low muscle mass/function (According to EWGSOP2 Criteria) | Uses dedicated screening model |
| **4. Osteosarcopenia** | Combined conditions (Low Bone Mass and Sarcopenia)| Predicts all 5 conditions at once |


---

### How to Use

1. **Select a page** from the sidebar on the left
2. **View the performance metrics** of the 15 available models
3. **Choose a model** for your prediction
4. **Enter patient data** in the form
5. **Get your prediction** with probability score

---

### About the Models

Each model was trained using:
- **Hybrid Feature Selection** (RFECV + RFE)
- **Cross-validated threshold optimization**
- **SMOTE-ENN / RandomUnderSampler** for class balancing

Performance metrics are based on held-out test set evaluation.

---

*Use the sidebar to navigate to a prediction page.*
""")

# Sidebar info
st.sidebar.success("Select a page above to start.")
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Data Source**")
st.sidebar.markdown("Models were trained using data from the Bushehr Elderly Health (BEH) Program (n = 4,110).")
st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ Disclaimer**")
st.sidebar.markdown("This tool is for research purposes only. Not for clinical diagnosis.")
