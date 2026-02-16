# src/fingerprint_styler.py
import cv2
import numpy as np
import os
from skimage import exposure, filters
import random

class FingerprintStyler:
    """
    Transforms R307 digital output to look like real fingerprint images.
    """
    
    def __init__(self, reference_images_path='../data/reference_fingerprints/'):
        self.reference_images_path = reference_images_path
        self.reference_images = []
        self.load_reference_images()
    
    def load_reference_images(self):
        """Load real fingerprint images for style reference."""
        if os.path.exists(self.reference_images_path):
            for f in os.listdir(self.reference_images_path):
                if f.endswith(('.png', '.bmp', '.jpg')):
                    img = cv2.imread(os.path.join(self.reference_images_path, f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (96, 96))
                        self.reference_images.append(img)
        
        # If no reference images, create synthetic ones
        if not self.reference_images:
            self.create_synthetic_references()
    
    def create_synthetic_references(self):
        """Create synthetic fingerprint patterns as reference."""
        for _ in range(5):
            img = np.zeros((96, 96), dtype=np.uint8)
            center = 48
            
            # Create fingerprint-like patterns
            for r in range(10, 45, 5):
                cv2.ellipse(img, (center, center), (r, r-2), 0, 0, 360, 200, 1)
                cv2.ellipse(img, (center-2, center-2), (r+1, r-1), 10, 0, 360, 150, 1)
            
            # Add minutiae
            for _ in range(10):
                x = random.randint(20, 75)
                y = random.randint(20, 75)
                cv2.circle(img, (x, y), 1, 255, -1)
            
            # Add noise
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            
            self.reference_images.append(img)
    
    def apply_style_transfer(self, r307_image):
        """
        Transform R307 digital output to look like real fingerprint.
        """
        # Ensure image is uint8
        if r307_image.dtype != np.uint8:
            r307_image = cv2.normalize(r307_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Step 1: Enhance contrast to make ridges visible
        enhanced = cv2.equalizeHist(r307_image)
        
        # Step 2: Apply adaptive thresholding to get ridge pattern
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Step 3: Apply morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Step 4: Create gradient-based ridge enhancement
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Step 5: Combine binary and gradient for natural look
        combined = cv2.addWeighted(binary, 0.6, gradient, 0.4, 0)
        
        # Step 6: Apply Gabor filter for ridge enhancement
        g_kernel = cv2.getGaborKernel((5,5), 4.0, np.pi/4, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor = cv2.filter2D(enhanced, cv2.CV_8UC3, g_kernel)
        
        # Step 7: Blend with reference style if available
        if self.reference_images:
            ref = random.choice(self.reference_images)
            
            # Match histogram to reference
            matched = exposure.match_histograms(combined, ref)
            matched = matched.astype(np.uint8)
        else:
            matched = combined
        
        # Step 8: Add realistic noise texture
        texture = np.random.normal(0, 5, matched.shape).astype(np.int16)
        textured = np.clip(matched.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        
        # Step 9: Final enhancement
        final = cv2.bilateralFilter(textured, 5, 50, 50)
        
        return final
    
    def batch_style_transfer(self, r307_images):
        """Apply style transfer to multiple images."""
        return [self.apply_style_transfer(img) for img in r307_images]


class FingerprintAugmenter:
    """
    Creates multiple variations of captured fingerprints for better training.
    """
    
    @staticmethod
    def augment_fingerprint(img):
        """Create realistic variations of the fingerprint."""
        variations = []
        
        # Original
        variations.append(img)
        
        # Slight rotation
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), random.uniform(-5, 5), 1)
        rotated = cv2.warpAffine(img, M, (cols, cols))
        variations.append(rotated)
        
        # Small translation
        M = np.float32([[1, 0, random.randint(-3, 3)], 
                        [0, 1, random.randint(-3, 3)]])
        translated = cv2.warpAffine(img, M, (cols, cols))
        variations.append(translated)
        
        # Scale variation
        scale = random.uniform(0.95, 1.05)
        scaled = cv2.resize(img, None, fx=scale, fy=scale)
        if scaled.shape[0] > cols:
            scaled = scaled[:cols, :cols]
        else:
            scaled = cv2.resize(scaled, (cols, cols))
        variations.append(scaled)
        
        # Add noise
        noisy = img.copy()
        noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
        noisy = cv2.add(noisy, noise)
        variations.append(noisy)
        
        return variations


# Test the styler
def test_styler():
    """Test the fingerprint styler."""
    from hardware_capture import FingerprintSensor
    
    sensor = FingerprintSensor('/dev/cu.usbserial-1420')
    styler = FingerprintStyler()
    
    if not sensor.connected:
        print("❌ Sensor not connected")
        return
    
    print("\n📸 Capturing R307 fingerprint...")
    r307_img = sensor.capture_fingerprint_image()
    
    if r307_img is not None:
        print("🎨 Applying style transfer...")
        styled = styler.apply_style_transfer(r307_img)
        
        # Display comparison
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(r307_img, cmap='gray')
        axes[0].set_title('R307 Original')
        axes[0].axis('off')
        
        axes[1].imshow(styled, cmap='gray')
        axes[1].set_title('Style Transferred')
        axes[1].axis('off')
        
        # Show a reference fingerprint
        if styler.reference_images:
            axes[2].imshow(styler.reference_images[0], cmap='gray')
            axes[2].set_title('Reference Style')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('../data/style_comparison.png')
        plt.show()
        
        # Save both
        cv2.imwrite('../data/r307_original.png', r307_img)
        cv2.imwrite('../data/r307_styled.png', styled)
        
        print("✅ Images saved")
    
    sensor.close()


if __name__ == "__main__":
    test_styler()