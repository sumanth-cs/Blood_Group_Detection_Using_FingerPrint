import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from hardware_capture import FingerprintSensor
import threading
import time

class BloodGroupApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Blood Group Screening System")
        self.root.geometry("900x600")
        self.root.configure(bg='#2c3e50')
        
        # Load trained model
        try:
            self.model = tf.keras.models.load_model('../trained_model/fingerprint_model.h5')
            print("Model loaded successfully")
        except:
            messagebox.showerror("Error", "Could not load AI model. Please train the model first.")
            self.model = None
        
        # Initialize sensor (use None for demo mode)
        self.sensor = None
        try:
            self.sensor = FingerprintSensor('/dev/cu.usbserial-1420')  # Update port
        except:
            print("Sensor not connected. Running in DEMO MODE.")
        
        # Blood group labels
        self.blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        self.colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9', 
                      '#9b59b6', '#8e44ad', '#2ecc71', '#27ae60']
        
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#34495e', height=80)
        header.pack(fill='x')
        
        title = tk.Label(header, text="🧬 AI-Based Blood Group Screening System", 
                        font=('Arial', 24, 'bold'), bg='#34495e', fg='white')
        title.pack(pady=20)
        
        disclaimer = tk.Label(header, 
                            text="⚠️ FOR DEMONSTRATION ONLY - NOT FOR MEDICAL USE", 
                            font=('Arial', 10), bg='#34495e', fg='#f1c40f')
        disclaimer.pack()
        
        # Main Content Frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left Panel - Image Display
        left_panel = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        tk.Label(left_panel, text="Fingerprint Scanner", font=('Arial', 16, 'bold'),
                bg='#34495e', fg='white').pack(pady=10)
        
        self.image_label = tk.Label(left_panel, bg='black', width=40, height=20)
        self.image_label.pack(padx=10, pady=10)
        
        # Right Panel - Controls and Results
        right_panel = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        right_panel.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        
        tk.Label(right_panel, text="System Control", font=('Arial', 16, 'bold'),
                bg='#34495e', fg='white').pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(right_panel, bg='#34495e')
        btn_frame.pack(pady=20)
        
        self.capture_btn = tk.Button(btn_frame, text="📷 Capture Fingerprint", 
                                    command=self.capture_fingerprint,
                                    font=('Arial', 12), bg='#3498db', fg='white',
                                    width=20, height=2)
        self.capture_btn.pack(pady=5)
        
        self.demo_btn = tk.Button(btn_frame, text="🔧 Demo Mode", 
                                 command=self.demo_mode,
                                 font=('Arial', 12), bg='#9b59b6', fg='white',
                                 width=20, height=2)
        self.demo_btn.pack(pady=5)
        
        # Results Display
        results_frame = tk.Frame(right_panel, bg='#2c3e50', relief='sunken', bd=2)
        results_frame.pack(pady=20, padx=10, fill='x')
        
        tk.Label(results_frame, text="AI Prediction Results", 
                font=('Arial', 14, 'bold'), bg='#2c3e50', fg='white').pack(pady=10)
        
        self.result_label = tk.Label(results_frame, text="Waiting for scan...", 
                                    font=('Arial', 18), bg='#2c3e50', fg='white')
        self.result_label.pack(pady=10)
        
        self.confidence_label = tk.Label(results_frame, text="Confidence: --%", 
                                        font=('Arial', 14), bg='#2c3e50', fg='#bdc3c7')
        self.confidence_label.pack(pady=5)
        
        # Confidence Bar
        self.progress = ttk.Progressbar(results_frame, length=200, 
                                       mode='determinate', maximum=100)
        self.progress.pack(pady=10)
        
        # Blood Group Distribution Chart Placeholder
        chart_frame = tk.Frame(right_panel, bg='#2c3e50')
        chart_frame.pack(pady=20)
        
        tk.Label(chart_frame, text="Prediction Distribution", 
                font=('Arial', 12), bg='#2c3e50', fg='white').pack()
        
        self.chart_canvas = tk.Canvas(chart_frame, width=300, height=150, bg='#2c3e50')
        self.chart_canvas.pack()
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def capture_fingerprint(self):
        """Capture fingerprint from sensor."""
        self.capture_btn.config(state='disabled', text="Capturing...")
        self.root.update()
        
        def capture_thread():
            try:
                if self.sensor:
                    # Actual hardware capture
                    img = self.sensor.capture_image()
                else:
                    # Demo fallback
                    img = cv2.imread('../data/sample_fingerprint.bmp', cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError("No sample image found")
                
                # Process and predict
                self.process_and_predict(img)
                
            except Exception as e:
                messagebox.showerror("Capture Error", f"Error: {str(e)}\nUsing demo data.")
                self.demo_mode()
            finally:
                self.capture_btn.config(state='normal', text="📷 Capture Fingerprint")
        
        threading.Thread(target=capture_thread, daemon=True).start()
    
    def demo_mode(self):
        """Use a pre-loaded sample for demonstration."""
        sample_path = '../data/sample_fingerprint.bmp'
        img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self.process_and_predict(img)
        else:
            messagebox.showwarning("Demo Mode", "Please place a sample_fingerprint.bmp in data folder")
    
    def process_and_predict(self, img):
        """Process image and run AI prediction."""
        # Display original image
        display_img = cv2.resize(img, (320, 240))
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        
        # Preprocess for model
        processed = cv2.resize(img, (96, 96))
        processed = processed.astype('float32') / 255.0
        processed = np.expand_dims(processed, axis=(0, -1))  # Add batch and channel dims
        
        # Predict
        if self.model:
            predictions = self.model.predict(processed, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx] * 100
            
            # Update results
            blood_group = self.blood_groups[predicted_idx]
            self.result_label.config(text=f"Predicted: {blood_group}", 
                                   fg=self.colors[predicted_idx])
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            self.progress['value'] = confidence
            
            # Update chart
            self.draw_prediction_chart(predictions)
            
            # Show explanation
            self.show_explanation(blood_group, confidence)
        else:
            self.result_label.config(text="Model not loaded", fg='red')
    
    def draw_prediction_chart(self, predictions):
        """Draw a simple bar chart of predictions."""
        self.chart_canvas.delete("all")
        
        width = 300
        height = 150
        bar_width = 30
        spacing = 5
        
        for i, (pred, group, color) in enumerate(zip(predictions, self.blood_groups, self.colors)):
            bar_height = pred * 100
            x0 = i * (bar_width + spacing) + 10
            y0 = height - bar_height
            x1 = x0 + bar_width
            y1 = height - 20
            
            # Draw bar
            self.chart_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='white')
            
            # Draw label
            self.chart_canvas.create_text(x0 + bar_width/2, y1 + 10, 
                                        text=group, fill='white', font=('Arial', 8))
            
            # Draw percentage
            self.chart_canvas.create_text(x0 + bar_width/2, y0 - 10, 
                                        text=f"{pred*100:.0f}%", fill='white', font=('Arial', 8))
    
    def show_explanation(self, blood_group, confidence):
        """Show popup explanation of results."""
        if confidence > 80:
            explanation = f"High confidence prediction: {blood_group}\n\nThe AI detected patterns in the fingerprint ridge characteristics that strongly correlate with the {blood_group} blood type in the training data."
        elif confidence > 60:
            explanation = f"Moderate confidence prediction: {blood_group}\n\nFurther verification would be needed for clinical purposes."
        else:
            explanation = f"Low confidence prediction: {blood_group}\n\nThis demonstrates the limitation of the model with unfamiliar or poor-quality fingerprints."
        
        messagebox.showinfo("AI Analysis Explanation", explanation)

if __name__ == "__main__":
    root = tk.Tk()
    app = BloodGroupApp(root)
    root.mainloop()