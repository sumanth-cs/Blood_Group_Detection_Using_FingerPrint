# src/hardware_adapter.py
import cv2
import numpy as np
import time
import streamlit as st

class HardwareAdapter:
    """Adapter class to interface with R307 sensor or provide fallback."""
    
    def __init__(self):
        self.sensor = None
        self._init_sensor()
    
    def _init_sensor(self):
        """Initialize the actual hardware sensor if available."""
        try:
            from hardware_capture import FingerprintSensor
            self.sensor = FingerprintSensor('/dev/cu.usbserial-1420')
            st.sidebar.success("✅ Hardware sensor connected")
        except Exception as e:
            st.sidebar.warning("⚠️ Hardware not connected - Using demo mode")
            self.sensor = None
    
    def capture(self):
        """Capture fingerprint from hardware or return synthetic image."""
        if self.sensor:
            try:
                return self.sensor.capture_image()
            except:
                return self._get_synthetic_image()
        else:
            return self._get_synthetic_image()
    
    def _get_synthetic_image(self):
        """Generate a synthetic fingerprint-like image for demo."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        # Add some patterns
        for i in range(0, 200, 15):
            cv2.line(img, (i, 0), (i + 80, 200), 200, 2)
            cv2.line(img, (0, i), (200, i + 40), 200, 2)
        return img
    
    def close(self):
        if self.sensor:
            self.sensor.close()