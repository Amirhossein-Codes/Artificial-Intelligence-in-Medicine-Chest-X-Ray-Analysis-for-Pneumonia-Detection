# Chest X-Ray Pneumonia Classifier

An AI-powered application that uses a pretrained deep learning model to classify chest X-ray images as either **Normal** or **Pneumonia**. The application features a modern, user-friendly GUI built with Flet for easy image upload and classification.

## Features

- 🏥 **Medical Image Classification**: Classifies chest X-ray images as Normal or Pneumonia
- 🖥️ **Modern GUI**: Beautiful and intuitive user interface built with Flet
- 📁 **Easy File Upload**: Drag and drop or browse to upload X-ray images
- 📊 **Detailed Results**: Shows prediction confidence and probabilities for both classes
- ⚡ **Fast Processing**: Optimized image preprocessing and model inference
- 🔄 **Real-time Feedback**: Progress indicators and status updates during classification

## Screenshots

The application provides:
- Clean, professional interface
- Image preview after upload
- Detailed classification results with confidence scores
- Color-coded results (Green for Normal, Red for Pneumonia)

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux

### Setup

1. **Clone or download the project files** to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model file is present**:
   - Make sure `best_chest_xray_model.h5` is in the project directory

## Usage

### Running the Application

1. **Start the GUI application**:
   ```bash
   python gui_app.py
   ```

2. **Wait for model loading**:
   - The application will show a loading message while initializing the AI model
   - Once loaded, you'll see a success notification

3. **Upload an image**:
   - Click the "📁 Upload X-Ray Image" button
   - Select a chest X-ray image (JPG, JPEG, PNG, or BMP format)
   - The image will be displayed in the preview area

4. **Classify the image**:
   - Click the "🔍 Classify Image" button
   - Wait for the analysis to complete
   - View the results showing the prediction and confidence scores

### Supported Image Formats

- JPG/JPEG
- PNG
- BMP

### Understanding Results

The application provides:
- **Prediction**: The classified result (Normal or Pneumonia)
- **Confidence**: How certain the model is about the prediction (0-100%)
- **Probabilities**: Detailed breakdown showing the likelihood for each class

## Project Structure

```
proProject/
├── best_chest_xray_model.h5    # Pretrained model file
├── chest_xray_classifier.py    # Core classification logic
├── gui_app.py                  # Flet GUI application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Technical Details

### Model Architecture

The application uses a pretrained deep learning model that:
- Accepts 224x224 pixel RGB images
- Outputs binary classification (Normal vs Pneumonia)
- Provides confidence scores for predictions

### Image Preprocessing

Images are automatically:
- Resized to 224x224 pixels
- Converted to RGB format
- Normalized to [0, 1] range
- Prepared for model inference

### Performance

- Model loading: ~5-10 seconds (first run)
- Image classification: ~1-3 seconds per image
- Memory usage: ~500MB-1GB (depending on model size)

## Troubleshooting

### Common Issues

1. **Model loading fails**:
   - Ensure `best_chest_xray_model.h5` is in the project directory
   - Check that TensorFlow is properly installed
   - Verify sufficient disk space and memory

2. **Image upload issues**:
   - Use supported formats (JPG, PNG, BMP)
   - Ensure image file is not corrupted
   - Check file permissions

3. **Classification errors**:
   - Ensure the image is a valid chest X-ray
   - Try different image formats
   - Check console for detailed error messages

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space
- **GPU**: Optional but recommended for faster processing

## Medical Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the documentation
- Enhancing the UI/UX

## License

This project is provided as-is for educational purposes.

## Support

If you encounter any issues or have questions, please check the troubleshooting section above or create an issue in the project repository.
