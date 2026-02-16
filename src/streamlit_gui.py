import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import time

# Page configuration
st.set_page_config(
    page_title="AI Blood Group Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-top: 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #ffc107;
        margin: 1rem 0;
    }
    .result-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNCTION DEFINITIONS
# ============================================
# Add this function to your streamlit_gui.py
def capture_and_display():
    """Enhanced capture function for Streamlit."""
    
    # Initialize hardware
    from hardware_capture import FingerprintSensor
    sensor = FingerprintSensor('/dev/cu.usbserial-1420')
    
    if not sensor.connected:
        st.error("Sensor not connected. Using demo mode.")
        return
    
    with st.spinner("Waiting for finger..."):
        result = sensor.capture_for_display()
    
    if result:
        # Create tabs to show different views
        tab1, tab2, tab3 = st.tabs(["Raw Capture", "Enhanced (AI-Ready)", "Edge Detection"])
        
        with tab1:
            st.image(result['raw'], caption="Raw Sensor Output", use_container_width=True)
            st.info("This is what the sensor sees - notice the 'spiral' pattern")
        
        with tab2:
            st.image(result['enhanced'], caption="Enhanced for AI", use_container_width=True)
            st.success("After enhancement - now matches training data quality!")
        
        with tab3:
            st.image(result['edges'], caption="Edge Detection", use_container_width=True)
            st.info("AI focuses on these ridge patterns")
        
        # Now use the enhanced image for prediction
        process_and_predict(result['for_ai'], model, blood_groups, group_colors, results_placeholder)
    
    sensor.close()
    
def display_results(predictions, blood_groups, group_colors, results_placeholder):
    """Display prediction results in the results container."""
    
    predicted_idx = np.argmax(predictions)
    predicted_group = blood_groups[predicted_idx]
    confidence = predictions[predicted_idx] * 100
    
    with results_placeholder.container():
        # Result box
        st.markdown(f"""
        <div class="result-box" style="background-color: {group_colors[predicted_group]}20; border: 2px solid {group_colors[predicted_group]};">
            <h2 style="color: {group_colors[predicted_group]}; margin: 0;">{predicted_group}</h2>
            <p style="font-size: 1.5rem; margin: 0;">{confidence:.1f}% Confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(int(confidence) / 100, text="Prediction Confidence")
        
        # Create metrics row
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Top Prediction", predicted_group, f"{confidence:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            second_best = np.argsort(predictions)[-2]
            st.metric("Second Best", blood_groups[second_best], 
                     f"{predictions[second_best]*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_c:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence Spread", f"{confidence:.1f}%", 
                     f"{(confidence - predictions[second_best]*100):.1f}% gap")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create interactive bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=blood_groups,
                y=predictions * 100,
                marker_color=list(group_colors.values()),
                text=[f"{p*100:.1f}%" for p in predictions],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
            )
        ])
        
        fig.update_traces(
            marker_line_color='black',
            marker_line_width=2,
            opacity=0.8
        )
        
        fig.update_layout(
            title={
                'text': "Prediction Distribution Across Blood Groups",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Blood Group",
            yaxis_title="Confidence (%)",
            yaxis_range=[0, 100],
            height=400,
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_yaxes(gridcolor='lightgray', gridwidth=1)
        fig.update_xaxes(gridcolor='lightgray', gridwidth=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation based on confidence
        if confidence > 80:
            st.success(f"""
            **High Confidence Prediction: {predicted_group}**  
            The AI detected strong patterns in the fingerprint ridge characteristics 
            that correlate with the {predicted_group} blood type in the training data.
            """)
        elif confidence > 60:
            st.info(f"""
            **Moderate Confidence Prediction: {predicted_group}**  
            The AI found some matching patterns, but with moderate certainty. 
            This demonstrates the need for larger training datasets.
            """)
        else:
            st.warning(f"""
            **Low Confidence Prediction: {predicted_group}**  
            The AI is uncertain about this prediction. This could be due to:
            - Poor image quality
            - Unusual fingerprint patterns
            - Limited training data for this blood group
            """)
        
        # Prediction details
        with st.expander("📋 Detailed Prediction Report"):
            df = pd.DataFrame({
                'Blood Group': blood_groups,
                'Confidence (%)': [f"{p*100:.2f}%" for p in predictions],
                'Raw Score': predictions
            })
            df = df.sort_values('Raw Score', ascending=False)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def process_and_predict(img, model, blood_groups, group_colors, results_placeholder):
    """Process image and display results."""
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return
    
    # Preprocess image
    processed = cv2.resize(img, (96, 96))
    processed = processed.astype('float32') / 255.0
    processed = np.expand_dims(processed, axis=(0, -1))
    
    # Make prediction
    with st.spinner("AI is analyzing the fingerprint..."):
        predictions = model.predict(processed, verbose=0)[0]
    
    # Display results
    display_results(predictions, blood_groups, group_colors, results_placeholder)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        # Try multiple possible paths
        possible_paths = [
            '../trained_model/fingerprint_model.h5',
            '../../trained_model/fingerprint_model.h5',
            './trained_model/fingerprint_model.h5',
            '/Users/revanthcs/Downloads/My_Projects/Blood_Group_Detection_Using_FingerPrint/trained_model/fingerprint_model.h5'
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            model = tf.keras.models.load_model(model_path)
            st.sidebar.success(f"✅ Model loaded from {os.path.basename(model_path)}")
            return model
        else:
            st.sidebar.error("❌ Model not found! Please train the model first.")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None

# ============================================
# MAIN APP STARTS HERE
# ============================================

# Title and header
st.markdown('<h1 class="main-header">🧬 AI-Based Blood Group Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Using Fingerprint Analysis with Deep Learning</p>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ FOR DEMONSTRATION ONLY - NOT FOR MEDICAL USE</strong><br>
    This is a proof-of-concept for educational purposes. Blood group determination 
    for medical purposes MUST be performed using standard serological tests.
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

# Blood group configuration
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
group_colors = {
    'A+': '#e74c3c', 'A-': '#c0392b',
    'B+': '#3498db', 'B-': '#2980b9',
    'AB+': '#9b59b6', 'AB-': '#8e44ad',
    'O+': '#2ecc71', 'O-': '#27ae60'
}

# Sidebar for controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/fingerprint-scan.png", width=100)
    st.title("🖐️ Control Panel")
    
    # Input method selection
    input_method = st.radio(
        "Choose Input Method",
        ["📤 Upload Fingerprint Image", "📸 Use Sample Image", "🎥 Demo Mode"],
        help="Select how you want to provide the fingerprint"
    )
    
    st.markdown("---")
    
    # Model info
    if model:
        st.success("✅ Model ready")
    else:
        st.error("❌ Model not loaded")
    
    # Hardware status
    st.markdown("### 📡 Hardware Status")
    try:
        import importlib.util
        spec = importlib.util.find_spec("hardware_capture")
        if spec is not None:
            st.success("✅ Hardware module available")
        else:
            st.warning("⚠️ Hardware module not found (Demo mode only)")
    except:
        st.warning("⚠️ Hardware not available (Demo mode active)")
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("Validation Accuracy", "66.94%", delta="±2.5%")
    st.metric("Precision", "72.99%", delta="±3%")
    st.metric("Recall", "58.36%", delta="±3%")

# Main content area - Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📷 Fingerprint Scanner")
    
    # Image display area
    image_placeholder = st.empty()
    
    # Results placeholder in col2
    with col2:
        st.markdown("### 📊 Analysis Results")
        results_placeholder = st.empty()
        
        # Initialize with empty results
        with results_placeholder.container():
            st.info("👆 Upload a fingerprint and click 'Analyze' to see results")
            
            # Show placeholder chart
            fig = go.Figure(data=[
                go.Bar(
                    x=blood_groups,
                    y=[0] * 8,
                    marker_color=list(group_colors.values()),
                    text=[f"{0:.1f}%" for _ in range(8)],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Prediction Distribution",
                xaxis_title="Blood Group",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100],
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model information
            with st.expander("ℹ️ About the AI Model"):
                st.markdown("""
                **Model Architecture:** CNN (VGG-style)
                - 4 Convolutional Blocks
                - Batch Normalization
                - Dropout for regularization
                - Dense layers: 512 → 256 → 8
                
                **Training Data:** 4,893 fingerprint images
                **Validation Data:** 1,107 fingerprint images
                **Input Size:** 96 × 96 pixels (grayscale)
                """)
    
    # Upload or capture image based on selection
    if input_method == "📤 Upload Fingerprint Image":
        uploaded_file = st.file_uploader(
            "Choose a fingerprint image", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
            help="Upload a clear fingerprint image for analysis"
        )
        
        if uploaded_file is not None:
            # Read and display image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # FIXED: Use use_container_width=True instead of width=None
                image_placeholder.image(img, caption="Uploaded Fingerprint", use_container_width=True)
                
                # Process button
                if st.button("🔍 Analyze Fingerprint", type="primary", use_container_width=True):
                    process_and_predict(img, model, blood_groups, group_colors, results_placeholder)
            else:
                st.error("Could not read image. Please try another file.")
    
    elif input_method == "📸 Use Sample Image":
        # Try different sample image formats
        sample_paths = [
            '../data/sample_fingerprint.png',
            '../data/sample_fingerprint.bmp',
            '../data/sample_fingerprint.jpg',
            '../data/sample.bmp',
            '../../data/sample_fingerprint.png',
            # Create a synthetic image if none found
        ]
        
        img = None
        used_path = None
        for path in sample_paths:
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    used_path = path
                    break
        
        if img is not None:
            # FIXED: Use use_container_width=True
            image_placeholder.image(img, caption=f"Sample Fingerprint ({os.path.basename(used_path)})", use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔍 Analyze Sample", type="primary", use_container_width=True):
                    process_and_predict(img, model, blood_groups, group_colors, results_placeholder)
            with col_b:
                if st.button("🔄 Use Different Sample", use_container_width=True):
                    st.rerun()
        else:
            st.warning("No sample image found. Creating synthetic fingerprint...")
            
            # Create a simple synthetic image
            synthetic_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
            # Add some ridge-like patterns
            for i in range(0, 200, 20):
                cv2.line(synthetic_img, (i, 0), (i + 100, 200), 200, 2)
                cv2.line(synthetic_img, (0, i), (200, i + 50), 200, 2)
            
            # FIXED: Use use_container_width=True
            image_placeholder.image(synthetic_img, caption="Synthetic Demo Fingerprint", use_container_width=True)
            
            if st.button("🔍 Analyze Synthetic", type="primary", use_container_width=True):
                process_and_predict(synthetic_img, model, blood_groups, group_colors, results_placeholder)
    
    else:  # Demo Mode
        st.info("🎥 Demo Mode Active - Using synthetic fingerprint data")
        
        # Create synthetic fingerprint for demo
        synthetic_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        # Add some ridge-like patterns
        for i in range(0, 200, 20):
            cv2.line(synthetic_img, (i, 0), (i + 100, 200), 200, 2)
            cv2.line(synthetic_img, (0, i), (200, i + 50), 200, 2)
        
        # FIXED: Use use_container_width=True
        image_placeholder.image(synthetic_img, caption="Demo Fingerprint (Synthetic)", use_container_width=True)
        
        if st.button("🎲 Generate Random Prediction", type="primary", use_container_width=True):
            with st.spinner("AI is analyzing..."):
                time.sleep(1.5)
                # Generate random predictions for demo
                demo_predictions = np.random.dirichlet(np.ones(8), size=1)[0]
                display_results(demo_predictions, blood_groups, group_colors, results_placeholder)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <p>🔬 Final Year Project - AI-Based Non-Invasive Blood Group Detection</p>
    <p>⚠️ This is a research prototype. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)