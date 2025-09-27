import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


class ChestXrayClassifier:
    def __init__(self, model_path="best_chest_xray_model.h5"):
        """
        Initialize the chest X-ray classifier with a pretrained model.
        """

        self.model = None
        self.model_path = model_path
        self.class_names = ['Normal', 'Pneumonia']
        self.load_model()
    
    def load_model(self):
        """Load the pretrained model from the .h5 file."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            
            # Check model output shape
            output_shape = self.model.output_shape
            print(f"Model output shape: {output_shape}")
            

            if len(output_shape) == 2:
                num_classes = output_shape[1]
                print(f"Model has {num_classes} output classes")
                
                # Update class names if needed
                if num_classes == 1:
                    self.class_names = ['Normal', 'Pneumonia']
                    print("Using binary classification with single output")
                elif num_classes == 2:
                    self.class_names = ['Normal', 'Pneumonia']
                    print("Using binary classification with two outputs")
                else:
                    # Multi-class classification
                    self.class_names = [f'Class_{i}' for i in range(num_classes)]
                    print(f"Using multi-class classification with {num_classes} classes")
            else:
                print("Unexpected model output shape")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image ready for prediction
        """
        try:
            # Load and resize image to 224x224 (common size for medical imaging)
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            # Convert BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224
            img = cv2.resize(img, (224, 224))
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path):
        """
        Predict whether an image shows Normal or Pneumonia.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_img)
            
            # Handle different model output formats
            if len(predictions.shape) == 2:
                # Standard multi-class output
                pred_array = predictions[0]
            else:
                # Single value output
                pred_array = predictions.flatten()
            
            print(f"Raw predictions: {pred_array}")
            print(f"Prediction shape: {predictions.shape}")
            
            # Handle different outputs
            if len(pred_array) == 1:
                # Binary classification with single output (0 = Normal, 1 = Pneumonia)
                probability = float(pred_array[0])
                
                if probability > 0.5:
                    predicted_class = 'Pneumonia'
                    confidence = probability
                else:
                    predicted_class = 'Normal'
                    confidence = 1.0 - probability
                
                probabilities = {
                    'Normal': 1.0 - probability,
                    'Pneumonia': probability
                }
                
            elif len(pred_array) == 2:
                # Binary classification with two outputs
                predicted_class_idx = np.argmax(pred_array)
                confidence = float(pred_array[predicted_class_idx])
                predicted_class = self.class_names[predicted_class_idx]
                
                probabilities = {
                    'Normal': float(pred_array[0]),
                    'Pneumonia': float(pred_array[1])
                }
                
            else:
                # Multi-class classification
                predicted_class_idx = np.argmax(pred_array)
                confidence = float(pred_array[predicted_class_idx])
                predicted_class = self.class_names[predicted_class_idx]
                
                # Create probabilities dict
                probabilities = {}
                for i, class_name in enumerate(self.class_names):
                    if i < len(pred_array):
                        probabilities[class_name] = float(pred_array[i])
                    else:
                        probabilities[class_name] = 0.0
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'class_index': predicted_class_idx if len(pred_array) > 1 else (0 if predicted_class == 'Normal' else 1)
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            print(f"Predictions shape: {predictions.shape if 'predictions' in locals() else 'Not available'}")
            raise
    
    def get_model_summary(self):
        """Get a summary of the loaded model."""
        if self.model:
            return str(self.model.summary())
        return "Model not loaded"

