# src/hardware_adapter.py
import cv2
import numpy as np
import time
import streamlit as st
from hardware_capture import FingerprintSensor

class HardwareAdapter:
    """Adapter class to interface with R307 sensor with Streamlit integration."""
    
    def __init__(self):
        self.sensor = None
        self.last_capture_time = 0
        self.cooldown = 2  # Seconds between captures
        self._init_sensor()
    
    def _init_sensor(self):
        """Initialize the actual hardware sensor if available."""
        try:
            self.sensor = FingerprintSensor('/dev/cu.usbserial-1420')
            if self.sensor.connected:
                st.sidebar.success("✅ R307 Sensor Connected")
            else:
                st.sidebar.warning("⚠️ Sensor detected but not responding")
                self.sensor = None
        except Exception as e:
            st.sidebar.warning(f"⚠️ Hardware not connected - Using demo mode: {str(e)[:50]}")
            self.sensor = None
    
    def capture(self, wait_for_finger=True, timeout=10):
        """
        Capture fingerprint with exhibition-friendly features.
        
        Returns:
            tuple: (raw_image, display_image, edges_image) or (None, None, None)
        """
        current_time = time.time()
        
        # Respect cooldown to prevent excessive captures
        if current_time - self.last_capture_time < self.cooldown:
            time.sleep(0.5)
        
        if self.sensor and self.sensor.connected:
            try:
                # Use the enhanced capture method
                raw_img, display_img, edges_img = self.sensor.capture_with_preview()
                self.last_capture_time = time.time()
                
                if raw_img is not None:
                    return raw_img, display_img, edges_img
                else:
                    return self._get_demo_images()
                    
            except Exception as e:
                st.error(f"Capture error: {str(e)}")
                return self._get_demo_images()
        else:
            return self._get_demo_images()
    
    def _get_demo_images(self):
        """Generate demo images for exhibition."""
        raw_img = self.sensor._get_demo_image() if self.sensor else self._create_synthetic()
        
        # Create display versions
        display_img = cv2.resize(raw_img, (320, 240))
        
        # Create edge-detected version
        edges = cv2.Canny(raw_img, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_colored[edges > 0] = [0, 255, 0]
        
        return raw_img, display_img, edges_colored
    
    def _create_synthetic(self):
        """Create synthetic fingerprint pattern."""
        img = np.zeros((288, 256), dtype=np.uint8)
        center_y, center_x = 144, 128
        
        for r in range(10, 200, 15):
            cv2.circle(img, (center_x, center_y), r, 200, 2)
        
        for angle in range(0, 360, 30):
            x = int(center_x + 100 * np.cos(np.radians(angle)))
            y = int(center_y + 100 * np.sin(np.radians(angle)))
            cv2.line(img, (center_x, center_y), (x, y), 150, 1)
        
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def get_status(self):
        """Get current hardware status."""
        if self.sensor and self.sensor.connected:
            return "connected", "✅ Live capture ready"
        else:
            return "demo", "🎥 Demo mode (no hardware)"
    
    def close(self):
        if self.sensor:
            self.sensor.close()