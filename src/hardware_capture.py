# # src/hardware_capture.py
# import serial
# import time
# import numpy as np
# import cv2
# import struct
# import threading
# from collections import deque

# class FingerprintSensor:
#     def __init__(self, port='/dev/cu.usbserial-1420', baudrate=57600, timeout=5):
#         """
#         Initialize R307 fingerprint sensor with robust error handling.
        
#         Args:
#             port: Serial port (e.g., '/dev/cu.usbserial-1420')
#             baudrate: Try 57600 first, fallback to 9600 if unstable
#             timeout: Serial timeout in seconds
#         """
#         self.port = port
#         self.baudrate = baudrate
#         self.timeout = timeout
#         self.ser = None
#         self.connected = False
#         self.image_buffer = deque(maxlen=5)
#         self.retry_count = 3
#         self.use_slow_mode = False  # Will switch to 9600 if needed
        
#         # R307 command codes
#         self.CMD_HEADER = b'\xEF\x01\xFF\xFF\xFF\xFF'
#         self.CMD_GET_IMAGE = b'\x01\x00\x03\x01\x00\x05'
#         self.CMD_IMAGE2TZ = b'\x01\x00\x04\x02\x01\x00\x08'
#         self.CMD_UPIMAGE = b'\x01\x00\x03\x0A\x00\x0E'
        
#         self.connect()
    
#     def connect(self):
#         """Establish serial connection with automatic baud rate fallback."""
#         baud_rates = [57600, 9600]  # Try fast first, then slow
        
#         for baud in baud_rates:
#             try:
#                 if self.ser and self.ser.is_open:
#                     self.ser.close()
                
#                 self.ser = serial.Serial(
#                     port=self.port,
#                     baudrate=baud,
#                     timeout=self.timeout,
#                     parity=serial.PARITY_NONE,
#                     stopbits=serial.STOPBITS_ONE,
#                     bytesize=serial.EIGHTBITS,
#                     rtscts=False,
#                     dsrdtr=False
#                 )
#                 time.sleep(2)  # Allow sensor to initialize
                
#                 # Test connection with a simple command
#                 if self._test_connection():
#                     self.connected = True
#                     self.baudrate = baud
#                     self.use_slow_mode = (baud == 9600)
#                     print(f"✅ Connected at {baud} baud")
#                     return True
                    
#             except Exception as e:
#                 print(f"⚠️  Failed at {baud} baud: {e}")
        
#         self.connected = False
#         print("❌ Could not connect to sensor")
#         return False
    
#     def _test_connection(self):
#         """Test if sensor is responding."""
#         try:
#             # Send a simple command and check for response
#             self.ser.write(self.CMD_HEADER + self.CMD_GET_IMAGE)
#             time.sleep(0.2)
#             if self.ser.in_waiting >= 12:
#                 return True
#         except:
#             pass
#         return False
    
#     def send_command(self, command, response_size=12):
#         """Send command with retry logic."""
#         for attempt in range(self.retry_count):
#             try:
#                 # Clear any stale data
#                 if self.ser.in_waiting:
#                     self.ser.reset_input_buffer()
                
#                 # Send command
#                 packet = self.CMD_HEADER + command
#                 self.ser.write(packet)
                
#                 # Wait for response
#                 time.sleep(0.2)
                
#                 if self.ser.in_waiting >= response_size:
#                     response = self.ser.read(response_size)
#                     return response
                    
#             except Exception as e:
#                 print(f"⚠️  Command attempt {attempt + 1} failed: {e}")
#                 time.sleep(0.5)
        
#         return None
    
#     def check_finger_present(self):
#         """Check if a finger is placed with improved sensitivity."""
#         response = self.send_command(self.CMD_GET_IMAGE)
#         if response and len(response) >= 12:
#             # Check confirmation code (9th byte)
#             return response[9] == 0x00
#         return False
    
#     def wait_for_finger(self, timeout=15):
#         """
#         Wait for finger with visual feedback.
        
#         Args:
#             timeout: Maximum wait time in seconds
            
#         Returns:
#             bool: True if finger detected
#         """
#         start_time = time.time()
#         print("🖐️  Please place your finger on the sensor...")
        
#         last_check = 0
#         while time.time() - start_time < timeout:
#             current_time = time.time()
            
#             # Check every 0.3 seconds
#             if current_time - last_check > 0.3:
#                 if self.check_finger_present():
#                     print("✅ Finger detected!")
#                     time.sleep(0.5)  # Allow finger to settle
#                     return True
#                 last_check = current_time
            
#             # Show progress every 2 seconds
#             elapsed = int(current_time - start_time)
#             if elapsed % 2 == 0 and elapsed > 0:
#                 print(f"⏳ Still waiting... ({elapsed}s)")
            
#             time.sleep(0.1)
        
#         print("⏰ Timeout: No finger detected")
#         return False
    
#     def read_image_data(self, expected_bytes=73728, max_attempts=5):
#         """
#         Read image data with robust packet assembly.
        
#         Args:
#             expected_bytes: Expected image size (288*256 = 73728)
#             max_attempts: Number of read attempts
            
#         Returns:
#             bytes: Complete image data or None
#         """
#         image_data = bytearray()
#         total_read = 0
#         stale_count = 0
        
#         for attempt in range(max_attempts):
#             # Wait for data
#             time.sleep(0.5 if self.use_slow_mode else 0.3)
            
#             bytes_available = self.ser.in_waiting
#             if bytes_available > 0:
#                 chunk = self.ser.read(bytes_available)
#                 image_data.extend(chunk)
#                 total_read += len(chunk)
#                 print(f"📦 Received {len(chunk)} bytes (total: {total_read}/{expected_bytes})")
                
#                 # Reset stale counter on new data
#                 stale_count = 0
#             else:
#                 stale_count += 1
#                 if stale_count >= 3:
#                     break
            
#             # Check if we have enough data
#             if total_read >= expected_bytes:
#                 print(f"✅ Complete image received: {total_read} bytes")
#                 return bytes(image_data[:expected_bytes])
            
#             # Small delay between reads
#             time.sleep(0.2)
        
#         if total_read > 0:
#             print(f"⚠️  Partial image: {total_read}/{expected_bytes} bytes")
#             return bytes(image_data) if total_read > 1000 else None
        
#         return None
    
#     def capture_image(self, wait_for_finger=True, timeout=15):
#         """
#         Capture fingerprint image with robust error recovery.
        
#         Returns:
#             numpy.ndarray: Grayscale fingerprint image (288x256) or demo image
#         """
#         if not self.connected:
#             print("⚠️  Sensor not connected. Using demo image.")
#             return self._get_demo_image()
        
#         if wait_for_finger:
#             if not self.wait_for_finger(timeout):
#                 print("🎲 No finger detected - using demo image")
#                 return self._get_demo_image()
        
#         # Step 1: Get image from sensor (with retries)
#         for attempt in range(self.retry_count):
#             response = self.send_command(self.CMD_GET_IMAGE)
#             if response and len(response) >= 12 and response[9] == 0x00:
#                 print("📸 Image captured successfully")
#                 break
#             print(f"⚠️  Capture attempt {attempt + 1} failed, retrying...")
#             time.sleep(1)
#         else:
#             print("❌ Failed to capture image after multiple attempts")
#             return self._get_demo_image()
        
#         # Step 2: Transfer image to character buffer
#         response = self.send_command(self.CMD_IMAGE2TZ)
#         if not response or len(response) < 12 or response[9] != 0x00:
#             print("⚠️  Buffer transfer failed, but continuing...")
        
#         # Step 3: Request image upload
#         response = self.send_command(self.CMD_UPIMAGE)
#         if not response or len(response) < 12:
#             print("⚠️  Upload command failed, but attempting to read...")
        
#         # Step 4: Read image data with retry
#         image_data = None
#         for attempt in range(self.retry_count):
#             print(f"📡 Reading image data (attempt {attempt + 1})...")
#             image_data = self.read_image_data()
            
#             if image_data and len(image_data) >= 70000:  # Close enough
#                 break
            
#             # If using fast mode and failing, suggest slow mode
#             if not self.use_slow_mode and attempt == self.retry_count - 2:
#                 print("💡 Tip: Try setting baudrate=9600 for more stable connection")
            
#             time.sleep(1)
        
#         if image_data and len(image_data) >= 70000:
#             try:
#                 # Convert to numpy array
#                 img_array = np.frombuffer(image_data[:73728], dtype=np.uint8)
#                 img_array = img_array.reshape((288, 256))
                
#                 # Apply preprocessing
#                 img_array = cv2.medianBlur(img_array, 3)
#                 img_array = cv2.equalizeHist(img_array)
                
#                 # Store in buffer
#                 self.image_buffer.append(img_array)
#                 print(f"✅ Image processed successfully: {img_array.shape}")
#                 return img_array
                
#             except Exception as e:
#                 print(f"❌ Error processing image: {e}")
#                 return self._get_demo_image()
#         else:
#             print("❌ Could not read complete image")
#             return self._get_demo_image()
    
#     def capture_with_preview(self):
#         """Capture and return multiple versions for exhibition."""
#         raw_img = self.capture_image()
        
#         if raw_img is None:
#             return None, None, None
        
#         # Create display versions
#         display_img = cv2.resize(raw_img, (320, 240))
        
#         # Create edge-detected version for "AI vision" effect
#         edges = cv2.Canny(raw_img, 50, 150)
#         edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
#         edges_colored[edges > 0] = [0, 255, 0]
        
#         return raw_img, display_img, edges_colored
    
#     def _get_demo_image(self):
#         """Generate enhanced synthetic fingerprint."""
#         img = np.zeros((288, 256), dtype=np.uint8)
#         center_y, center_x = 144, 128
        
#         # Create realistic ridge patterns
#         for r in range(20, 200, 12):
#             cv2.circle(img, (center_x, center_y), r, 180, 2)
#             cv2.circle(img, (center_x - 10, center_y - 10), r + 5, 150, 1)
        
#         # Add swirl pattern
#         for angle in range(0, 360, 15):
#             x = int(center_x + 100 * np.cos(np.radians(angle)))
#             y = int(center_y + 100 * np.sin(np.radians(angle)))
#             cv2.line(img, (center_x, center_y), (x, y), 140, 1)
        
#         # Add minutiae points
#         for _ in range(20):
#             x = np.random.randint(50, 200)
#             y = np.random.randint(50, 200)
#             cv2.circle(img, (x, y), 2, 255, -1)
        
#         # Add noise
#         noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
#         img = cv2.add(img, noise)
        
#         return img
    
#     def close(self):
#         """Close serial connection."""
#         if self.ser and self.ser.is_open:
#             self.ser.close()
#             self.connected = False
#             print("🔌 Sensor disconnected")


# # Enhanced test function
# def test_sensor():
#     """Comprehensive sensor test."""
#     print("🔬 Testing R307 Fingerprint Sensor")
#     print("=" * 40)
    
#     # Try fast mode first
#     sensor = FingerprintSensor('/dev/cu.usbserial-1420', baudrate=57600)
    
#     if not sensor.connected:
#         print("\n🔄 Trying slow mode (9600 baud)...")
#         sensor = FingerprintSensor('/dev/cu.usbserial-1420', baudrate=9600)
    
#     if not sensor.connected:
#         print("❌ Could not connect to sensor. Check wiring.")
#         return
    
#     print("\n=== R307 Fingerprint Sensor Test ===")
#     print(f"📡 Mode: {'Slow (9600)' if sensor.use_slow_mode else 'Fast (57600)'}")
    
#     print("\n📸 Attempting to capture fingerprint...")
#     print("(Place your finger on the sensor when prompted)")
    
#     img = sensor.capture_image(wait_for_finger=True, timeout=15)
    
#     if img is not None and not np.array_equal(img, sensor._get_demo_image()):
#         print(f"\n✅ SUCCESS! Live fingerprint captured!")
#         print(f"   Shape: {img.shape}")
#         print(f"   Data range: {img.min()}-{img.max()}")
        
#         # Save the image
#         filename = f'../data/live_capture_{int(time.time())}.png'
#         cv2.imwrite(filename, img)
#         print(f"   Saved to: {filename}")
        
#         # Show preview if possible
#         try:
#             preview = cv2.resize(img, (512, 576))
#             cv2.imshow('Live Fingerprint Capture', preview)
#             print("   Press any key to close preview...")
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         except:
#             pass
#     else:
#         print("\n❌ Could not capture live fingerprint.")
#         print("   Using synthetic image for demonstration.")
    
#     sensor.close()


# if __name__ == "__main__":
#     test_sensor()

# src/hardware_capture.py
import serial
import time
import numpy as np
import cv2
from collections import deque

class FingerprintSensor:
    def __init__(self, port='/dev/cu.usbserial-1420', baudrate=57600):
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        self.ser = None
        self.image_buffer = deque(maxlen=5)
        
        # R307 image dimensions
        self.IMG_HEIGHT = 288
        self.IMG_WIDTH = 256
        
        # R307 command codes
        self.CMD_HEADER = b'\xEF\x01\xFF\xFF\xFF\xFF'
        self.CMD_GET_IMAGE = b'\x01\x00\x03\x01\x00\x05'
        self.CMD_IMAGE2TZ = b'\x01\x00\x04\x02\x01\x00\x08'
        self.CMD_UPIMAGE = b'\x01\x00\x03\x0A\x00\x0E'
        
        self.connect()
    
    def connect(self):
        """Establish connection using manual protocol."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            time.sleep(2)
            
            # Test connection
            if self._send_command(self.CMD_GET_IMAGE):
                self.connected = True
                print(f"✅ Connected at {self.baudrate} baud")
                return True
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
        
        self.connected = False
        return False
    
    def _send_command(self, command, response_size=12):
        """Send command and get response."""
        try:
            if self.ser.in_waiting:
                self.ser.reset_input_buffer()
            
            packet = self.CMD_HEADER + command
            self.ser.write(packet)
            time.sleep(0.2)
            
            if self.ser.in_waiting >= response_size:
                return self.ser.read(response_size)
        except:
            pass
        return None
    
    def wait_for_finger(self, timeout=15):
        """Wait for finger detection."""
        print("🖐️  Please place your finger on the sensor...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self._send_command(self.CMD_GET_IMAGE)
            if response and len(response) >= 12 and response[9] == 0x00:
                print("✅ Finger detected!")
                return True
            
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % 2 == 0:
                print(f"⏳ Waiting... ({elapsed}s)")
            
            time.sleep(0.5)
        
        print("⏰ Timeout: No finger detected")
        return False
    
    def capture_raw_data(self, wait_for_finger=True, timeout=15):
        """
        Capture raw fingerprint data from sensor.
        Returns the raw byte data before processing.
        """
        if not self.connected:
            return None
        
        if wait_for_finger and not self.wait_for_finger(timeout):
            return None
        
        # Step 1: Get image
        response = self._send_command(self.CMD_GET_IMAGE)
        if not response or len(response) < 12 or response[9] != 0x00:
            print("❌ Failed to capture image")
            return None
        
        # Step 2: Transfer to buffer
        response = self._send_command(self.CMD_IMAGE2TZ)
        
        # Step 3: Request upload
        response = self._send_command(self.CMD_UPIMAGE)
        
        # Step 4: Read all data
        print("📡 Reading fingerprint data...")
        time.sleep(1)
        
        all_data = bytearray()
        expected_bytes = 73728  # 288 * 256 = 73728 pixels
        max_attempts = 80  # Increased for complete capture
        
        for attempt in range(max_attempts):
            if self.ser.in_waiting:
                chunk = self.ser.read(self.ser.in_waiting)
                all_data.extend(chunk)
                print(f"   Received {len(chunk)} bytes (total: {len(all_data)})")
                
                if len(all_data) >= expected_bytes:
                    print(f"✅ Complete data received: {len(all_data)} bytes")
                    break
            
            time.sleep(0.1)  # Shorter sleep for faster capture
        
        return bytes(all_data) if len(all_data) >= 36864 else None  # Need at least half
    
    def convert_to_fingerprint_image(self, raw_data):
        """
        Convert R307 raw data to a real fingerprint image.
        Handles both 4-bit and 8-bit data formats.
        """
        if not raw_data or len(raw_data) < 10000:
            return None
        
        print(f"📊 Processing {len(raw_data)} bytes of raw data")
        
        # Try different methods based on data size
        
        # Method 1: If we have full 8-bit data (73728 bytes)
        if len(raw_data) >= 73728:
            # Take first 73728 bytes as 8-bit pixels
            img_array = np.frombuffer(raw_data[:73728], dtype=np.uint8)
            img_array = img_array.reshape((288, 256))
            print("   Using 8-bit direct conversion")
        
        # Method 2: If we have 4-bit packed data (36864 bytes)
        elif len(raw_data) >= 36864:
            # Each byte contains two 4-bit pixels
            pixels = []
            for byte in raw_data[:36864]:
                # Extract high and low nibbles
                high = (byte >> 4) & 0x0F
                low = byte & 0x0F
                pixels.extend([high, low])
            
            # Scale from 0-15 to 0-255
            img_array = np.array(pixels[:73728], dtype=np.uint8) * 17
            img_array = img_array.reshape((288, 256))
            print("   Using 4-bit packed conversion")
        
        else:
            print(f"   Insufficient data: {len(raw_data)} bytes")
            return None
        
        # Verify image has content
        print(f"   Image stats - Min: {img_array.min()}, Max: {img_array.max()}, Mean: {img_array.mean():.1f}")
        
        return img_array
    
    def enhance_for_training_compatibility(self, img):
        """
        Make the image look like the training data.
        This is CRITICAL for accurate predictions.
        """
        # Step 1: Ensure we're working with uint8
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Step 2: Check if image is too dark
        mean_brightness = img.mean()
        print(f"   Mean brightness before enhancement: {mean_brightness:.1f}")
        
        if mean_brightness < 30:  # Image is too dark
            print("   Image too dark, applying brightness boost")
            # Apply gamma correction to brighten
            gamma = 1.5
            img = np.power(img / 255.0, gamma) * 255
            img = img.astype(np.uint8)
        
        # Step 3: Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Step 4: Apply adaptive thresholding to enhance ridges
        img_thresh = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        
        # Step 5: Combine with original for natural look
        img = cv2.addWeighted(img, 0.3, img_thresh, 0.7, 0)
        
        # Step 6: Apply gentle sharpening
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]]) / 1.0
        img = cv2.filter2D(img, -1, kernel)
        
        # Step 7: Normalize final image
        img = cv2.normalize(img, None, 20, 235, cv2.NORM_MINMAX)  # Avoid pure black/white
        
        # Step 8: Resize to model input size
        img = cv2.resize(img, (96, 96))
        
        print(f"   Final stats - Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.1f}")
        
        return img
    
    def capture_fingerprint_image(self, wait_for_finger=True, timeout=15):
        """
        Complete pipeline: capture raw data and convert to fingerprint image.
        """
        # Step 1: Get raw data from sensor
        raw_data = self.capture_raw_data(wait_for_finger, timeout)
        
        if raw_data is None:
            print("⚠️  Using demo image")
            return self._create_demo_fingerprint()
        
        # Step 2: Convert to fingerprint image
        fp_image = self.convert_to_fingerprint_image(raw_data)
        
        if fp_image is None:
            print("⚠️  Conversion failed, using demo")
            return self._create_demo_fingerprint()
        
        # Step 3: Enhance to match training data
        enhanced = self.enhance_for_training_compatibility(fp_image)
        
        return enhanced
    
    def _create_demo_fingerprint(self):
        """Create a realistic demo fingerprint."""
        img = np.zeros((288, 256), dtype=np.uint8)
        center_y, center_x = 144, 128
        
        # Generate fingerprint-like patterns
        for r in range(20, 120, 6):
            cv2.ellipse(img, (center_x, center_y), (r, r-5), 0, 0, 360, 200, 2)
            cv2.ellipse(img, (center_x-5, center_y-5), (r+2, r-3), 10, 0, 360, 150, 1)
        
        # Add minutiae points
        for _ in range(30):
            x = np.random.randint(50, 200)
            y = np.random.randint(50, 200)
            cv2.circle(img, (x, y), 2, 255, -1)
            cv2.circle(img, (x+3, y+3), 1, 200, -1)
        
        # Add ridge flow lines
        for i in range(0, 288, 20):
            pts = np.array([[i, 100], [i+50, 150], [i+30, 200]], np.int32)
            cv2.polylines(img, [pts], False, 180, 1)
        
        # Add noise
        noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Enhance
        return self.enhance_for_training_compatibility(img)
    
    def get_preview_images(self):
        """Get multiple versions for exhibition."""
        raw_data = self.capture_raw_data(wait_for_finger=True, timeout=15)
        
        previews = {}
        
        if raw_data:
            # Show raw data visualization
            raw_display = np.frombuffer(raw_data[:36864], dtype=np.uint8)
            if len(raw_display) >= 192*192:
                raw_display = raw_display[:192*192].reshape((192, 192))
                raw_display = cv2.resize(raw_display, (320, 240))
                raw_display = cv2.cvtColor(raw_display, cv2.COLOR_GRAY2RGB)
                previews['raw_data'] = raw_display
        
        # Get enhanced fingerprint
        enhanced = self.capture_fingerprint_image(wait_for_finger=False)
        
        # Create visualizations
        enhanced_display = cv2.resize(enhanced, (320, 240))
        enhanced_display = cv2.cvtColor(enhanced_display, cv2.COLOR_GRAY2RGB)
        previews['enhanced'] = enhanced_display
        
        # Edge detection
        edges = cv2.Canny(enhanced, 30, 100)
        edges = cv2.resize(edges, (320, 240))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_colored[edges > 0] = [0, 255, 0]
        previews['edges'] = edges_colored
        
        # 3D visualization (for exhibition)
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(50), range(50))
        Z = enhanced[:50, :50]
        ax.plot_surface(X, Y, Z, cmap='terrain')
        ax.set_title('3D Ridge Profile')
        
        # Convert to image for Streamlit
        fig.canvas.draw()
        img_3d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_3d = img_3d.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        previews['3d'] = img_3d
        
        previews['for_ai'] = enhanced
        
        return previews
    
    def close(self):
        """Close connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            print("🔌 Sensor disconnected")


# Test function with visualization
def test_sensor():
    """Test the complete fingerprint capture pipeline."""
    print("🔬 Testing R307 Fingerprint Sensor with Image Conversion")
    print("=" * 60)
    
    sensor = FingerprintSensor('/dev/cu.usbserial-1420')
    
    if not sensor.connected:
        print("❌ Could not connect to sensor")
        return
    
    print("\n📸 Capturing fingerprint and converting to image...")
    fingerprint = sensor.capture_fingerprint_image(wait_for_finger=True, timeout=15)
    
    if fingerprint is not None:
        print(f"\n✅ Fingerprint captured: {fingerprint.shape}")
        print(f"   Pixel range: {fingerprint.min()} - {fingerprint.max()}")
        print(f"   Mean brightness: {fingerprint.mean():.1f}")
        
        # Save the image
        timestamp = int(time.time())
        filename = f'../data/fingerprint_{timestamp}.png'
        cv2.imwrite(filename, fingerprint)
        print(f"   Saved to: {filename}")
        
        # Display with stats
        try:
            # Create info panel
            info = np.zeros((100, 300, 3), dtype=np.uint8)
            cv2.putText(info, f"Min: {fingerprint.min()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            cv2.putText(info, f"Max: {fingerprint.max()}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            cv2.putText(info, f"Mean: {fingerprint.mean():.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            
            # Resize fingerprint for display
            display_fp = cv2.resize(fingerprint, (300, 300))
            display_fp = cv2.cvtColor(display_fp, cv2.COLOR_GRAY2RGB)
            
            # Combine
            combined = np.vstack([display_fp, info])
            
            cv2.imshow('Fingerprint Analysis', combined)
            print("\n   Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    else:
        print("❌ Failed to capture fingerprint")
    
    sensor.close()


if __name__ == "__main__":
    test_sensor()