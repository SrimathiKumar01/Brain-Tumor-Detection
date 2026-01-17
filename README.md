# Brain Tumor Detection System

This advanced brain tumor detection system utilizes deep learning technology to analyze MRI scans and identify potential tumors. The system provides both detection and visualization capabilities to assist in the analysis of brain MRI scans.

## Features

- 🔍 **Tumor Detection**: Analyzes MRI scans using a deep learning model.
- 🎯 **Region Visualization**: Highlights potential tumor regions.
- 📊 **Analysis History**: Tracks and visualizes detection history.
- 📈 **Detailed Metrics**: Provides confidence scores and image properties.

## How It Works

1. Upload an MRI scan image.
2. The system processes the image using advanced computer vision techniques.
3. A deep learning model analyzes the processed image.
4. Results are displayed with confidence scores and visualizations.

## Best Practices

- Use clear, high-quality MRI scans.
- Ensure images are properly oriented.
- Use standard medical imaging formats.
- Regular system calibration and validation.

## Technical Details

- **Model**: Custom CNN architecture.
- **Image Processing**: OpenCV.
- **Visualization**: Plotly & Streamlit.
- **Data Analysis**: Pandas & NumPy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Brain-Tumor-Detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Brain-Tumor-Detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the model file `brain_tumor_detector.h5` is in the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and go to `http://localhost:8501`.

## System Information

- **Python Version**: 3.8+
- **TensorFlow Version**: 2.15.0
- **OpenCV Version**: 4.9.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Developed by The Care Crew for medical imaging analysis.
