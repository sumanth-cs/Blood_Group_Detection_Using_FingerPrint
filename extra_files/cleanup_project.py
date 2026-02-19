# save as cleanup_project.py
import os
import shutil
from pathlib import Path

def cleanup_project():
    """Move unnecessary files to archive folder."""
    
    # Create archive folder
    archive_dir = Path('old_files_backup')
    archive_dir.mkdir(exist_ok=True)
    
    # Files to archive (not needed for final demo)
    files_to_archive = [
        'src/data_preprocessing.py',
        'src/preprocess_data.py',
        'src/prepare_dataset_final.py',
        'src/presentation_gui.py',
        'src/train_gender_model.py',
        'src/train_gender_model_balanced.py',
        'src/train_gender_model_fixed.py',
        'src/convert_gender_to_r307.py',
        'src/prepare_gender_data.py',
        'src/debug_sensor.py',
        'src/cleanup_old_files.py',
        'src/fingerprint_styler.py',
        'src/hardware_adapter.py',
        'src/hardware_capture.py',  # Using simple_hardware.py instead
        'src/test_hardware.py',
        'src/utils.py',
        'src/r307_simulator.py',
        'run_demo.sh',
    ]
    
    # Move files
    for file_path in files_to_archive:
        if os.path.exists(file_path):
            dest = archive_dir / os.path.basename(file_path)
            shutil.move(file_path, dest)
            print(f"📦 Archived: {file_path}")
    
    print(f"\n✅ Cleanup complete! Files moved to {archive_dir}")
    print("\n📁 Your working files should be:")
    print("   - app/exhibition_app.py")
    print("   - app/simple_hardware.py")
    print("   - models/*.h5")
    print("   - data/ (keep as is)")

if __name__ == "__main__":
    # Run from project root
    os.chdir(os.path.dirname(__file__))
    cleanup_project()