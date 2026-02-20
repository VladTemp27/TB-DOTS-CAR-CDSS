import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="TB-DOTS CAR CDSS",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 TB-DOTS CAR Data Visualization")
st.subheader("Tuberculosis Data Exploration & Analysis Prototype")

st.warning("⚠️ **Prototype Notice**: This is a visualization prototype for exploratory data analysis only. It is not intended for clinical decision-making or patient care.")

st.markdown("---")

# Overview Section
st.markdown("""
### Welcome

This prototype provides interactive visualizations and exploratory data analysis 
of Tuberculosis (TB) patient data from the **Cordillera Administrative Region (CAR)**, Philippines.

The system consolidates data from multiple health facilities to enable data exploration 
and trend analysis for research and planning purposes.
""")

# Key Metrics Overview
col1, col2 = st.columns(2)

with col1:
    st.info("""
    **📊 Non-Temporal Dataset**
    - Regional TB surveillance data
    - 2015-2025 records
    - Multiple provinces & municipalities
    """)

with col2:
    st.info("""
    **📈 Temporal Dataset**  
    - 4 Health facilities in Baguio City
    - Monthly treatment adherence tracking
    - 2016-2025 patient records
    """)

st.markdown("---")

# Features Section
st.markdown("### 🔧 Visualization Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Exploration**
    - Interactive charts & graphs
    - Demographic breakdowns
    - Treatment outcome distributions
    - Filterable data tables
    """)

with col2:
    st.markdown("""
    **Trend Analysis**
    - Year-over-year comparisons
    - Facility-level summaries
    - Treatment adherence tracking
    - Co-morbidity distributions
    """)

st.markdown("---")

# Navigation Guide
st.markdown("### 📍 Getting Started")
st.markdown("""
Use the **sidebar** to navigate between pages:

1. **📊 Dataset** - Explore and visualize TB patient data
2. **🤖 Model** - View model information (under development)

""")

# Footer
st.markdown("---")
st.caption("TB-DOTS CAR Visualization Prototype • Built with Streamlit • For research and educational purposes only")
