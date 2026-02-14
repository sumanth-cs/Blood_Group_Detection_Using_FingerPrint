#!/bin/bash
# run_demo.sh - Launch the Streamlit app

# Activate environment
source ~/tfenv/bin/activate

# Navigate to project
cd ~/Desktop/AI_BloodGroup_Project/src

# Launch Streamlit
streamlit run streamlit_gui.py --server.port 8501 --server.address localhost