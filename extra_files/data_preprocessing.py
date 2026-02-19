import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance
import random

def augment_image(image):
    """Apply random augmentations to simulate data variety."""
    pil_img = Image.fromarray(image)
    
    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    pil_img = pil_img.rotate(angle, fillcolor=0)
    
    # Random brightness
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Random contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    return np.array(pil_img)

def process_and_save(source_dir, target_dir, img_size=(96, 96), augment=True):
    """Reads, processes, and saves images to the target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.bmp')):
            img_path = os.path.join(source_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize
                img = cv2.resize(img, img_size)
                # Normalize to [0, 1]
                img = img.astype('float32') / 255.0
                # Save original processed image
                base_name = os.path.splitext(filename)[0]
                np.save(os.path.join(target_dir, f"{base_name}_orig.npy"), img)
                
                if augment:
                    # Create 4 augmented versions
                    for i in range(4):
                        aug_img = augment_image((img * 255).astype('uint8'))
                        aug_img = aug_img.astype('float32') / 255.0
                        np.save(os.path.join(target_dir, f"{base_name}_aug{i}.npy"), aug_img)

# Example usage - Run this for each blood group folder
# process_and_save('raw_fingerprints/A+', 'data/train/A+')