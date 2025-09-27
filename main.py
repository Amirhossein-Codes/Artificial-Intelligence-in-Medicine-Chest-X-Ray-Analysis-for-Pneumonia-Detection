import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
from chest_xray_classifier import ChestXrayClassifier

class ChestXrayGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chest X-Ray Pneumonia Classifier")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.classifier = None
        self.current_image_path = None
        self.prediction_result = None
        self.image_tk = None  
        
        # Create UI
        self.create_widgets()
        
        # Initialize classifier in background
        self.initialize_classifier()
    
    def create_widgets(self):
        """Create all UI widgets."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Chest X-Ray Pneumonia Classifier",
            font=('Arial', 24, 'bold'),
            foreground='#2c3e50'
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Upload a chest X-ray image to classify as Normal or Pneumonia",
            font=('Arial', 12),
            foreground='#7f8c8d'
        )
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Upload button
        self.upload_button = ttk.Button(
            main_frame,
            text="📁 Upload X-Ray Image",
            command=self.upload_image,
            style='Accent.TButton'
        )
        self.upload_button.grid(row=2, column=0, columnspan=3, pady=(0, 20))
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=3, column=0, columnspan=3, pady=(0, 20), sticky=(tk.W, tk.E))
        image_frame.columnconfigure(0, weight=1)
        
        # Image label
        self.image_label = ttk.Label(
            image_frame,
            text="No image selected",
            font=('Arial', 12),
            foreground='#95a5a6'
        )
        self.image_label.grid(row=0, column=0, pady=50)
        
        # Image info
        self.image_info_label = ttk.Label(
            main_frame,
            text="",
            font=('Arial', 10),
            foreground='#7f8c8d'
        )
        self.image_info_label.grid(row=4, column=0, columnspan=3, pady=(0, 10))
        
        # Classify button
        self.classify_button = ttk.Button(
            main_frame,
            text="🔍 Classify Image",
            command=self.classify_image,
            state='disabled'
        )
        self.classify_button.grid(row=5, column=0, columnspan=3, pady=(0, 20))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=300
        )
        self.progress_bar.grid(row=6, column=0, columnspan=3, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Ready to classify images",
            font=('Arial', 10),
            foreground='#7f8c8d'
        )
        self.status_label.grid(row=7, column=0, columnspan=3, pady=(0, 20))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="15")
        results_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E))
        results_frame.columnconfigure(0, weight=1)
        
        # Results content
        self.results_text = tk.Text(
            results_frame,
            height=8,
            width=50,
            font=('Arial', 10),
            wrap=tk.WORD,
            state='disabled',
            bg='#f8f9fa',
            relief='flat'
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        # Configure styles
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 11, 'bold'))
    
    def initialize_classifier(self):
        """Initialize the classifier in a background thread."""
        def load_model():
            try:
                self.classifier = ChestXrayClassifier()
                self.root.after(0, self.update_status, "Model loaded successfully! Ready to classify images.", "success")
            except Exception as e:
                self.root.after(0, self.update_status, f"Error loading model: {str(e)}", "error")
                self.root.after(0, messagebox.showerror, "Error", f"Failed to load model: {str(e)}")
        
        self.update_status("Loading AI model... Please wait.", "loading")
        self.progress_bar.start()
        
        # Run in background thread
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    
    def update_status(self, message, status_type="info"):
        """Update status message with appropriate styling."""
        self.status_label.config(text=message)
        
        # Set color based on status type
        if status_type == "success":
            self.status_label.config(foreground='#27ae60')
        elif status_type == "error":
            self.status_label.config(foreground='#e74c3c')
        elif status_type == "loading":
            self.status_label.config(foreground='#3498db')
        else:
            self.status_label.config(foreground='#7f8c8d')
        
        # Stop progress bar if not loading
        if status_type != "loading":
            self.progress_bar.stop()
    
    def upload_image(self):
        """Handle image upload."""
        file_path = filedialog.askopenfilename(
            title="Select X-Ray Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.update_image_info(file_path)
            
            if self.classifier:
                self.classify_button.config(state='normal')
                self.update_status("Image selected. Ready to classify.", "success")
            else:
                self.update_status("Image selected. Waiting for model to load...", "loading")
    
    def display_image(self, image_path):
        """Display the selected image."""
        try:
            # Load and resize image
            image = Image.open(image_path)
            
            # Calculate resize dimensions (max 400x300)
            max_width = 400
            max_height = 300
            
            # Get original dimensions
            width, height = image.size
            
            # Calculate scaling factor
            scale = min(max_width/width, max_height/height)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.image_tk = ImageTk.PhotoImage(image)
            
            # Update image label
            self.image_label.config(image=self.image_tk, text="")
            
        except Exception as e:
            self.image_label.config(image="", text=f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def update_image_info(self, file_path):
        """Update image information display."""
        try:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024  # KB
            
            info_text = f"Selected: {file_name} ({file_size:.1f} KB)"
            self.image_info_label.config(text=info_text)
            
        except Exception as e:
            self.image_info_label.config(text=f"Error getting file info: {str(e)}")
    
    def classify_image(self):
        """Classify the uploaded image."""
        if not self.current_image_path or not self.classifier:
            return
        
        # Disable button and show progress
        self.classify_button.config(state='disabled')
        self.progress_bar.start()
        self.update_status("Analyzing image...", "loading")
        
        def classify():
            try:
                result = self.classifier.predict(self.current_image_path)
                self.prediction_result = result
                
                # Update UI on main thread
                self.root.after(0, self.display_results)
                
            except Exception as e:
                self.root.after(0, self.show_error, str(e))
        
        # Run classification in background thread
        thread = threading.Thread(target=classify, daemon=True)
        thread.start()
    
    def display_results(self):
        """Display classification results."""
        if not self.prediction_result:
            return
        
        result = self.prediction_result
        
        # Determine styling based on prediction
        if result['class'] == 'Normal':
            status_icon = "✅"
            status_color = "green"
        else:
            status_icon = "⚠️"
            status_color = "red"
        
        # Create results text
        results_text = f"""
{status_icon} PREDICTION: {result['class'].upper()}

Confidence: {result['confidence']:.2%}

PROBABILITIES:
• Normal: {result['probabilities']['Normal']:.2%}
• Pneumonia: {result['probabilities']['Pneumonia']:.2%}

Analysis completed successfully!
        """.strip()
        
        # Update results display
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
        self.results_text.config(state='disabled')
        
        # Update status
        self.progress_bar.stop()
        self.classify_button.config(state='normal')
        self.update_status("Classification complete!", "success")
        
        # Show result in message box
        messagebox.showinfo(
            "Classification Result",
            f"Prediction: {result['class']}\nConfidence: {result['confidence']:.2%}"
        )
    
    def show_error(self, error_message):
        """Show error message."""
        self.progress_bar.stop()
        self.classify_button.config(state='normal')
        self.update_status(f"Error: {error_message}", "error")
        
        messagebox.showerror("Classification Error", f"Failed to classify image: {error_message}")

def main():
    root = tk.Tk()
    app = ChestXrayGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
