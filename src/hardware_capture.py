import serial
import time
import numpy as np
import cv2

class FingerprintSensor:
    def __init__(self, port='/dev/cu.usbserial-1420', baudrate=57600):
        """Initialize serial connection to R307 sensor."""
        self.ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(2)  # Allow time for initialization
        print(f"Connected to sensor on {port}")
    
    def capture_image(self):
        """Capture a fingerprint image and return as numpy array."""
        # Simplified command for image capture (R307 specific protocol)
        # Note: Actual implementation requires detailed protocol knowledge
        # This is a placeholder structure
        
        cmd_get_image = b'\xEF\x01\xFF\xFF\xFF\xFF\x01\x00\x03\x01\x00\x05'
        self.ser.write(cmd_get_image)
        time.sleep(1)
        
        # Read response (simplified - actual protocol is more complex)
        if self.ser.in_waiting:
            response = self.ser.read(1000)  # Adjust based on actual data
            # Process the byte data to extract image
            # This requires implementing the full R307 protocol
        
        # FOR DEMO: Load a sample image if hardware fails
        print("Hardware capture placeholder. Loading sample image.")
        sample_img = cv2.imread('../data/sample_fingerprint.bmp', cv2.IMREAD_GRAYSCALE)
        return sample_img
    
    def close(self):
        self.ser.close()

# For exhibition, have backup sample images ready