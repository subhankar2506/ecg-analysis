# ECG Arrhythmia Classification

![ECG Analysis](deploy/thumbnail.jpg)

## Overview

This project implements a deep learning model for classifying ECG signals from the MIT-BIH Arrhythmia Database into different types of arrhythmias. The model classifies ECG data into predefined categories of heart rhythms, such as normal rhythm and various arrhythmias (PVCs, APBs, AF, etc.).

**Live Streamlit App**: [BeatWise ECG Analysis App](https://beatwise.streamlit.app/)

## System Architecture

**Flowchart of the end-to-end system workflow**

The complete flowchart of our ECG arrhythmia classification system workflow can be viewed at the link below. Since the diagram contains detailed information about each component of the system, we recommend viewing it in a dedicated browser window where you can zoom and pan as needed.

[View complete system flowchart](https://www.mermaidchart.com/raw/5d761366-020f-4932-8a0a-e5ed457fa881?theme=light&version=v0.1&format=svg)

The flowchart illustrates the complete workflow from data acquisition through preprocessing, feature engineering, model development, and deployment to our Streamlit web application.

## Project Structure

```
ecg-analysis/
├── LICENSE
├── README.md
├── ecg_analysis.ipynb      # Main analysis notebook
├── deploy/                 # Deployment files
│   ├── app.py              # Streamlit app
│   ├── app_v1.py           # Previous version
│   ├── model.py            # Model definition
│   ├── requirements.txt    # Dependencies
│   ├── sample_ecg.csv      # Sample ECG data
│   ├── segments_label_encoder_classes.npy  # Label encoder
│   ├── thumbnail.jpg       # App thumbnail
│   └── xceptiontime_best.pth  # Trained model weights
├── images/                 # Visualization images
├── models/                 # Saved models
└── processed/              # Processed data
```

## Dataset

The project uses the MIT-BIH Arrhythmia Database, which contains:
- 48 half-hour excerpts of two-lead ECG recordings (lead I and lead II)
- Annotated arrhythmias in WFDB format (used by PhysioNet)
- Various arrhythmia types including:
  - Normal Sinus Rhythm (NSR)
  - Premature Ventricular Contractions (PVCs)
  - Atrial Premature Beats (APBs)
  - Atrial Fibrillation (AF)
  - And more

## Methodology

### 1. Data Preprocessing

- **ECG Signal Loading**: Used the `wfdb` library to read ECG signal data from the MIT-BIH database
- **Normalization**: Normalized the ECG signals to ensure consistent amplitude for model training
- **Labeling**: Used annotation files to label each ECG segment (e.g., normal, PVC, APB)

### 2. Feature Extraction

- **HRV (Heart Rate Variability)**: Measured RR intervals and extracted HRV-related features
- **Morphological Features**: Extracted features based on the shape of ECG waves (P-wave, QRS complex, T-wave)

### 3. Model Development

Our final model is based on XceptionTime architecture, which has shown excellent performance on time series classification tasks. The model was trained to classify ECG signals into multiple arrhythmia categories.

### 4. Model Performance Summary

The XceptionTime model combines the strengths of depthwise separable convolutions (from Xception) with multi-scale temporal processing to effectively classify ECG arrhythmias. The model architecture was specifically designed to:

1. Extract time-dependent features at multiple scales using dilated convolutions
2. Focus on the most important parts of the signal using attention mechanisms
3. Efficiently process the signal using depthwise separable convolutions
4. Learn robust representations through residual connections

The evaluation metrics show the model's performance on the MIT-BIH Arrhythmia Database. The model achieves strong results across different arrhythmia types, with particularly good performance on common arrhythmias.

### 5. Key advantages of this approach:

- **Efficiency**: The model uses depthwise separable convolutions which drastically reduce the number of parameters compared to standard convolutions
- **Attention mechanisms**: The model learns where to focus in the ECG signal through its global context attention module
- **Multi-scale processing**: Different dilation rates in the convolutional blocks allow the model to capture patterns at different time scales
- **Strong generalization**: Techniques like dropout, batch normalization, and residual connections help prevent overfitting

This model demonstrates state-of-the-art performance for ECG arrhythmia classification and could be valuable for clinical applications.

### 6. Model Evaluation

- Evaluated model performance using standard classification metrics
- Compared performance across various arrhythmia types
- Conducted cross-validation to ensure robustness

### 7. Results and Visualization

- Plotted results and metrics
- Visualized ECG signals with predictions, highlighting correct and incorrect classifications
- Visualized attention gradient of the XceptionTime model architecture

## Web Application

We've developed a user-friendly web interface using Streamlit that allows users to upload ECG files. The system processes the input and provides real-time predictions of the arrhythmia type based on the trained model.

### Features:
- Upload ECG files in various formats
- Real-time ECG signal visualization
- Arrhythmia classification with confidence scores
- Detailed explanation of detected arrhythmias

## Installation and Usage

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/subhankar2506/ecg-analysis.git
cd ecg-analysis
```

2. Create and activate a virtual environment:
```bash
cd deploy
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

### Using the Web App

1. Visit [https://beatwise.streamlit.app/](https://beatwise.streamlit.app/)
2. Upload an ECG file (either `.dat` and `.hea` files as per MIT-BIH format / `.csv` format)
3. View the visualization and classification results

## Future Enhancements

- Integration with wearable ECG devices
- Real-time monitoring capabilities
- Additional arrhythmia types
- Mobile application
- API for integration with other health systems

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in the repository.

## Acknowledgments

- MIT-BIH Arrhythmia Database for providing the ECG data
- PhysioNet for maintaining the healthcare datasets
- The open-source community for various libraries used in this project
