# src/test_hardware.py
from hardware_capture import test_sensor

if __name__ == "__main__":
    print("🔬 Testing R307 Fingerprint Sensor")
    print("=" * 40)
    test_sensor()