# Save as ~/Desktop/AI_BloodGroup_Project/src/prepare_dataset_final.py
import os
import cv2
import numpy as np
from PIL import Image
import shutil

def prepare_dataset():
    """Complete dataset preparation."""
    
    data_dir = '../data'
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    print("="*60)
    print("🔧 COMPLETE DATASET PREPARATION")
    print("="*60)
    
    total_bmp = 0
    total_dsstore = 0
    total_converted = 0
    total_corrupted = 0
    
    for split in ['train', 'validation']:
        for group in blood_groups:
            folder = os.path.join(data_dir, split, group)
            if not os.path.exists(folder):
                continue
                
            print(f"\n📁 Processing: {split}/{group}")
            
            # Remove .DS_Store
            ds_store = os.path.join(folder, '.DS_Store')
            if os.path.exists(ds_store):
                os.remove(ds_store)
                total_dsstore += 1
                print(f"  🗑️  Removed .DS_Store")
            
            # Process all files
            files = os.listdir(folder)
            bmp_files = [f for f in files if f.lower().endswith('.bmp')]
            
            for bmp_file in bmp_files:
                total_bmp += 1
                bmp_path = os.path.join(folder, bmp_file)
                
                try:
                    # Read image
                    img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        img = Image.open(bmp_path).convert('L')
                        img = np.array(img)
                    
                    # Save as PNG
                    png_file = bmp_file[:-4] + '.png'
                    png_path = os.path.join(folder, png_file)
                    cv2.imwrite(png_path, img)
                    
                    # Remove BMP
                    os.remove(bmp_path)
                    total_converted += 1
                    print(f"  ✅ {bmp_file:20} -> {png_file}")
                    
                except Exception as e:
                    total_corrupted += 1
                    print(f"  ❌ Corrupted: {bmp_file} - {str(e)}")
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY:")
    print(f"  🗑️  .DS_Store removed: {total_dsstore}")
    print(f"  🔄 BMP files found: {total_bmp}")
    print(f"  ✅ BMP converted to PNG: {total_converted}")
    print(f"  ⚠️  Corrupted files: {total_corrupted}")
    
    # Verify no BMP files remain
    remaining_bmp = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.bmp'):
                remaining_bmp.append(os.path.join(root, file))
    
    if remaining_bmp:
        print(f"\n❌ WARNING: {len(remaining_bmp)} BMP files still exist!")
        for bmp in remaining_bmp[:5]:
            print(f"     {bmp}")
    else:
        print("\n✅ SUCCESS: No BMP files remain. Your dataset is ready!")
        print("\n🚀 Now run: python train_model.py")
    
    return total_converted

if __name__ == "__main__":
    prepare_dataset()