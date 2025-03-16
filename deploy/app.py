# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import tempfile
import shutil
import wfdb
import datetime
import base64
from matplotlib.gridspec import GridSpec

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
from model import XceptionTime

# Set up page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Classification",
    page_icon="❤️",
    layout="wide"
)

# Function to load the trained model on CPU
@st.cache_resource
def load_model():
    # Get the number of classes
    class_names = load_class_names()
    num_classes = len(class_names)
    
    # Initialize model
    model = XceptionTime(
        input_channels=2,  # Two ECG leads
        num_classes=num_classes,
        initial_filters=64,
        depth=6,
        kernel_size=15,
        dropout=0.3
    )
    
    # Load model weights to CPU explicitly
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'xceptiontime_best.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Running in demo mode with random predictions.")
    
    model.eval()
    return model

# Function to load the class names
def load_class_names():
    try:
        # Try to load from saved file first
        current_dir = os.path.dirname(os.path.abspath(__file__))
        label_encoder_classes = np.load(os.path.join(current_dir, 'segments_label_encoder_classes.npy'))
        label_encoder_classes = np.load('segments_label_encoder_classes.npy')
        return list(label_encoder_classes)
    except:
        # Fall back to hardcoded classes if file not found
        return ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 
                'Bundle_Branch_Block', 'Paced', 'Unknown']

# Function to process WFDB format files
def process_wfdb_files(data_file, header_file=None, window_size=256):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # First, read the header file to determine the record name
        header_content = header_file.getvalue().decode('utf-8', errors='ignore')
        header_lines = header_content.strip().split('\n')
        
        # The first line of the header typically starts with the record name
        if header_lines and ' ' in header_lines[0]:
            # Extract the original record name
            original_record_name = header_lines[0].split(' ')[0]
            st.write(f"Original record name from header: {original_record_name}")
        else:
            # Fallback to a default name
            original_record_name = "record"
            st.write("Could not determine original record name, using default.")
        
        # Save files with the original record name to maintain consistency
        data_path = os.path.join(temp_dir, f"{original_record_name}.dat")
        header_path = os.path.join(temp_dir, f"{original_record_name}.hea")
        
        # Save data file
        with open(data_path, 'wb') as f:
            f.write(data_file.getbuffer())
        
        # Save header file
        with open(header_path, 'wb') as f:
            f.write(header_file.getbuffer())
        
        st.success("Files saved with original record name for consistency.")
        
        # Try to read the record
        record_path = os.path.join(temp_dir, original_record_name)
        try:
            record = wfdb.rdrecord(record_path)
            st.success("WFDB file successfully loaded!")
            
            # Convert to DataFrame for visualization
            signal_df = pd.DataFrame(record.p_signal, 
                                    columns=[f'Lead {i+1}' for i in range(record.p_signal.shape[1])])
            
            with st.expander("View Signal Data", expanded=False):
                st.write("First few rows of signal data:")
                st.write(signal_df.head())
            
            # Extract signals
            signals = record.p_signal
            
            # Check if we have at least 2 channels
            if signals.shape[1] < 2:
                st.error(f"The ECG file has only {signals.shape[1]} channel(s). We need at least 2.")
                return None, None
            
            # Normalize the signals (use only first 2 channels)
            normalized_signals = np.zeros((signals.shape[0], 2))
            for i in range(2):  # Use only first 2 channels
                normalized_signals[:, i] = (signals[:, i] - np.mean(signals[:, i])) / (np.std(signals[:, i]) + 1e-8)
            
            # Segment the signal
            segments = []
            for start in range(0, len(normalized_signals) - window_size + 1, window_size // 2):
                end = start + window_size
                if end <= len(normalized_signals):
                    segment = normalized_signals[start:end]
                    segments.append(segment)
            
            # If no complete segments, take whatever we can get
            if not segments and len(normalized_signals) > 0:
                segment = normalized_signals[:min(window_size, len(normalized_signals))]
                # Pad if necessary
                if segment.shape[0] < window_size:
                    pad_width = window_size - segment.shape[0]
                    segment = np.pad(segment, ((0, pad_width), (0, 0)), 'constant')
                segments.append(segment)
            
            if segments:
                st.success(f"Successfully extracted {len(segments)} segments from the ECG.")
            else:
                st.error("Could not extract any segments from the ECG.")
            
            return segments, record
        
        except Exception as e:
            st.error(f"Error reading WFDB record: {e}")
            
            # Check if we need to try a different approach
            try:
                # Try reading with wfdb.rdsamp which is sometimes more forgiving
                signals, fields = wfdb.rdsamp(record_path)
                st.success("Successfully read signals using rdsamp!")
                
                # Check if we have at least 2 channels
                if signals.shape[1] < 2:
                    st.error(f"The ECG file has only {signals.shape[1]} channel(s). We need at least 2.")
                    return None, None
                
                # Normalize the signals (use only first 2 channels)
                normalized_signals = np.zeros((signals.shape[0], 2))
                for i in range(2):  # Use only first 2 channels
                    normalized_signals[:, i] = (signals[:, i] - np.mean(signals[:, i])) / (np.std(signals[:, i]) + 1e-8)
                
                # Segment the signal
                segments = []
                for start in range(0, len(normalized_signals) - window_size + 1, window_size // 2):
                    end = start + window_size
                    if end <= len(normalized_signals):
                        segment = normalized_signals[start:end]
                        segments.append(segment)
                
                # If no complete segments, take whatever we can get
                if not segments and len(normalized_signals) > 0:
                    segment = normalized_signals[:min(window_size, len(normalized_signals))]
                    # Pad if necessary
                    if segment.shape[0] < window_size:
                        pad_width = window_size - segment.shape[0]
                        segment = np.pad(segment, ((0, pad_width), (0, 0)), 'constant')
                    segments.append(segment)
                
                if segments:
                    st.success(f"Successfully extracted {len(segments)} segments from the ECG using alternative method.")
                    # Create a dummy record for consistency
                    class DummyRecord:
                        def __init__(self, signals):
                            self.p_signal = signals
                            self.fs = 250  # Assumed sampling rate
                            self.record_name = original_record_name
                    
                    record = DummyRecord(signals)
                    return segments, record
                else:
                    st.error("Could not extract any segments from the ECG.")
                    return None, None
                
            except Exception as alt_e:
                st.error(f"Alternative method also failed: {alt_e}")
                return None, None
    
    except Exception as e:
        st.error(f"Error processing WFDB files: {e}")
        return None, None
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            st.warning(f"Warning: Could not clean up temporary directory: {e}")

# Function to process CSV format ECG data
def process_csv_ecg(file, window_size=256):
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check if the file has enough columns for at least 2 leads
        if df.shape[1] < 2:
            st.error("CSV file must have at least 2 columns (for 2 ECG leads).")
            return None, None
        
        # Use the first two columns as Lead I and Lead II
        signals = df.iloc[:, :2].values
        
        # Show data preview
        with st.expander("View CSV Data", expanded=False):
            st.write("First few rows of CSV data:")
            st.write(df.head())
            
            st.write("CSV data shape:", df.shape)
            
            if df.shape[0] > 1000:
                st.success(f"CSV contains {df.shape[0]} rows, which is sufficient for analysis.")
            else:
                st.warning(f"CSV contains only {df.shape[0]} rows. More data would provide better results.")
        
        # Normalize the signals
        normalized_signals = np.zeros_like(signals, dtype=float)
        for i in range(signals.shape[1]):
            normalized_signals[:, i] = (signals[:, i] - np.mean(signals[:, i])) / np.std(signals[:, i])
        
        # Segment the signal
        segments = []
        for start in range(0, len(normalized_signals) - window_size + 1, window_size // 2):
            end = start + window_size
            if end <= len(normalized_signals):
                segment = normalized_signals[start:end]
                segments.append(segment)
        
        # If no complete segments, take whatever we can get
        if not segments and len(normalized_signals) > 0:
            segment = normalized_signals[:min(window_size, len(normalized_signals))]
            # Pad if necessary
            if segment.shape[0] < window_size:
                pad_width = window_size - segment.shape[0]
                segment = np.pad(segment, ((0, pad_width), (0, 0)), 'constant')
            segments.append(segment)
            
        # Create a dummy record for consistency
        class DummyRecord:
            def __init__(self, signals):
                self.p_signal = signals
                self.fs = 250  # Assumed sampling rate
                self.record_name = "csv_data"
        
        record = DummyRecord(signals)
        
        return segments, record
    
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None, None

# Function to generate synthetic ECG data for demonstration
def generate_sample_ecg(window_size=256, num_segments=5, condition=None):
    segments = []
    
    # Generate synthetic ECG signal
    for i in range(num_segments):
        # Create time axis
        t = np.linspace(0, 2*np.pi, window_size)
        
        # Create a base signal that resembles ECG
        # P wave
        p_wave = 0.25 * np.sin(t + 0.2*i)
        
        # QRS complex
        qrs = np.zeros_like(t)
        qrs_idx = (t > 0.4*np.pi) & (t < 0.6*np.pi)
        
        # Modify QRS complex based on condition
        if condition == 'Ventricular':
            # Wider QRS for ventricular arrhythmia
            qrs_idx = (t > 0.4*np.pi) & (t < 0.7*np.pi)
            qrs[qrs_idx] = 1.8 * np.sin(2*(t[qrs_idx] - 0.5*np.pi))
        elif condition == 'Supraventricular':
            # Narrow QRS with abnormal P wave for supraventricular
            qrs[qrs_idx] = 1.2 * np.sin(3.5*(t[qrs_idx] - 0.5*np.pi))
            # Modify P wave
            p_wave = 0.4 * np.sin(1.5*t + 0.3*i)
        elif condition == 'Bundle_Branch_Block':
            # Wider QRS with notching for BBB
            qrs[qrs_idx] = 1.5 * np.sin(2.5*(t[qrs_idx] - 0.5*np.pi))
            # Add notching
            notch_idx = (t > 0.5*np.pi) & (t < 0.55*np.pi)
            qrs[notch_idx] *= 0.5
        else:
            # Normal QRS
            qrs[qrs_idx] = 1.5 * np.sin(3*(t[qrs_idx] - 0.5*np.pi))
        
        # T wave
        t_wave = 0.35 * np.sin(0.5*(t - 1.2*np.pi))
        t_wave[t < 0.7*np.pi] = 0
        
        # Combine components for lead I
        lead1 = p_wave + qrs + t_wave
        # Create lead II with slight variations
        lead2 = 0.8*p_wave + 1.2*qrs + 0.9*t_wave
        
        # Add some noise and variation between segments
        noise1 = np.random.normal(0, 0.05, window_size)
        noise2 = np.random.normal(0, 0.05, window_size)
        
        # Create the segment
        segment = np.column_stack((lead1 + noise1, lead2 + noise2))
        segments.append(segment)
    
    # Create a dummy record for consistency
    class DummyRecord:
        def __init__(self):
            self.p_signal = segments[0]  # Just use the first segment as example
            self.fs = 250  # Assumed sampling rate
            self.record_name = "synthetic_data"
    
    record = DummyRecord()
    
    return segments, record

# Function to display ECG preview
def display_ecg_preview(segment, record=None, title="ECG Signal Preview"):
    st.subheader(title)
    
    # Get time values if record has sampling frequency
    fs = 250  # Default value
    if record and hasattr(record, 'fs') and record.fs:
        fs = record.fs
    
    # Create time in seconds
    time_sec = np.arange(segment.shape[0]) / fs
    
    # Create figure with grid
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Add ECG grid background
    ax.grid(which='major', axis='both', linestyle='-', linewidth=0.5, color='#FF0000', alpha=0.2)
    ax.grid(which='minor', axis='both', linestyle='-', linewidth=0.5, color='#FF0000', alpha=0.1)
    ax.minorticks_on()
    
    # Normalize both leads to fit properly in the graph
    lead1 = segment[:, 0]
    lead2 = segment[:, 1]
    
    # Plot the signals
    ax.plot(time_sec, lead1, label='Lead I', linewidth=1.5)
    ax.plot(time_sec, lead2, label='Lead II', linewidth=1.5, alpha=0.7)
    
    # Ensure the y-axis limits are appropriate for the data
    y_min = min(np.min(lead1), np.min(lead2)) - 0.2  # Increased padding
    y_max = max(np.max(lead1), np.max(lead2)) + 0.2  # Increased padding
    
    # Set wider limits to ensure everything fits with some margin
    ax.set_ylim(y_min, y_max)
    
    # Add labels
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (normalized)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    # Add standard ECG scale indicator (0.5mV x 0.2s) in a better position
    # Place it in the middle-right of the plot
    scale_x = time_sec[-1] * 0.75  # 75% of the way across
    scale_y = y_min + (y_max - y_min) * 0.2  # 20% up from the bottom
    
    # Draw scale lines
    ax.plot([scale_x, scale_x], [scale_y, scale_y + 0.5], 'k-', linewidth=2)  # 0.5mV vertical line
    ax.plot([scale_x, scale_x + 0.2], [scale_y, scale_y], 'k-', linewidth=2)  # 0.2s horizontal line
    
    # Add scale labels with better positioning
    ax.text(scale_x, scale_y + 0.5 + 0.05, "0.5mV", ha='center', va='bottom', fontsize=8)
    ax.text(scale_x + 0.2 + 0.02, scale_y, "0.2s", ha='left', va='center', fontsize=8)
    
    # Display the plot
    st.pyplot(fig)
    
    # Add display options
    with st.expander("Display Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Signal Statistics:")
            stats_df = pd.DataFrame({
                'Lead I': {
                    'Mean': np.mean(segment[:, 0]),
                    'Std Dev': np.std(segment[:, 0]),
                    'Min': np.min(segment[:, 0]),
                    'Max': np.max(segment[:, 0])
                },
                'Lead II': {
                    'Mean': np.mean(segment[:, 1]),
                    'Std Dev': np.std(segment[:, 1]),
                    'Min': np.min(segment[:, 1]),
                    'Max': np.max(segment[:, 1])
                }
            })
            st.write(stats_df)
        
        with col2:
            st.write("Duration Information:")
            duration = segment.shape[0] / fs
            heartbeats = max(1, int(duration * 1.2))  # Estimate based on typical heart rate
            st.write(f"Signal duration: {duration:.2f} seconds")
            st.write(f"Sampling rate: {fs} Hz")
            st.write(f"Approx. heartbeats: ~{heartbeats}")

# Calculate basic ECG metrics
def calculate_ecg_metrics(segment, fs=250):
    """Calculate basic ECG metrics"""
    # Try to detect R peaks and calculate heart rate
    try:
        from scipy import signal as sp_signal
        
        # Process lead II (usually clearer for R peaks)
        lead_ii = segment[:, 1]
        
        # Filter to enhance QRS complexes
        filtered = sp_signal.butter(3, [5, 15], 'bandpass', fs=fs, output='sos')
        filtered = sp_signal.sosfilt(filtered, lead_ii)
        filtered = np.abs(filtered)
        
        # Find R peaks
        r_peaks, _ = sp_signal.find_peaks(filtered, height=0.5*np.max(filtered), distance=0.2*fs)
        
        # Calculate metrics
        metrics = {}
        
        if len(r_peaks) >= 2:
            # Calculate RR intervals
            rr_intervals = np.diff(r_peaks) / fs  # in seconds
            
            # Heart rate
            heart_rates = 60 / rr_intervals  # beats per minute
            metrics['heart_rate'] = np.mean(heart_rates)
            metrics['heart_rate_min'] = np.min(heart_rates)
            metrics['heart_rate_max'] = np.max(heart_rates)
            
            # Heart rate variability
            metrics['hrv_sdnn'] = np.std(rr_intervals)  # Standard deviation of NN intervals
            
            # RR interval statistics
            metrics['rr_mean'] = np.mean(rr_intervals)
            metrics['rr_std'] = np.std(rr_intervals)
            
            return metrics, r_peaks
        else:
            return {'heart_rate': 'Cannot calculate (insufficient R peaks)'}, None
    
    except Exception as e:
        return {'error': str(e)}, None

# Function to generate a PDF report
def generate_report(prediction, probabilities, metrics, segments):
    """Generate a simple PDF report using ReportLab"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        import io
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create a list to hold the flowables
        elements = []
        
        # Add title
        title_style = styles['Heading1']
        elements.append(Paragraph("ECG Arrhythmia Analysis Report", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add timestamp
        date_style = styles['Normal']
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elements.append(Paragraph(f"Generated on: {timestamp}", date_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add classification result
        result_style = styles['Heading2']
        elements.append(Paragraph("Classification Result", result_style))
        elements.append(Paragraph(f"Prediction: {prediction}", styles['Normal']))
        elements.append(Paragraph(f"Confidence: {probabilities[np.argmax(probabilities)]:.2%}", styles['Normal']))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add metrics
        if metrics:
            elements.append(Paragraph("ECG Measurements", result_style))
            data = [["Metric", "Value"]]
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    data.append([metric, f"{value:.2f}"])
                else:
                    data.append([metric, str(value)])
            
            # Create the table
            table = Table(data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.25*inch))
        
        # If segments exist, create and add an ECG image
        if segments and len(segments) > 0:
            # Plot the first segment
            segment = segments[0]
            fig, ax = plt.subplots(figsize=(6, 3))
            
            # Normalize to ensure good fit
            lead1 = segment[:, 0]
            lead2 = segment[:, 1]
            
            # Calculate appropriate y limits with more padding
            y_min = min(np.min(lead1), np.min(lead2)) - 0.2
            y_max = max(np.max(lead1), np.max(lead2)) + 0.2
            
            # Create x-axis values (samples)
            x = np.arange(len(lead1))
            
            # Plot with appropriate scaling
            ax.plot(x, lead1, label='Lead I')
            ax.plot(x, lead2, label='Lead II')
            ax.set_ylim(y_min, y_max)
            
            # Add scale indicator in a better position
            scale_x = len(x) * 0.75
            scale_y = y_min + (y_max - y_min) * 0.2
            
            # Draw scale lines (just for visual reference)
            ax.plot([scale_x, scale_x], [scale_y, scale_y + 0.5], 'k-', linewidth=2)
            ax.plot([scale_x, scale_x + 50], [scale_y, scale_y], 'k-', linewidth=2)  # 50 samples
            
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
            ax.set_title('ECG Signal')
            ax.legend()
            ax.grid(True)
            
            # Save the plot to a buffer
            img_buffer = io.BytesIO()
            plt.tight_layout()
            fig.savefig(img_buffer, format='png', dpi=150)
            img_buffer.seek(0)
            plt.close(fig)
            
            # Add the image to the PDF
            elements.append(Paragraph("ECG Signal", result_style))
            elements.append(Image(img_buffer, width=6*inch, height=3*inch))
        
        # Add disclaimer
        elements.append(Spacer(1, 0.5*inch))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            textColor=colors.red,
            fontSize=8
        )
        elements.append(Paragraph("DISCLAIMER: This analysis is provided for educational purposes only and should be confirmed by a healthcare professional.", disclaimer_style))
        
        # Build the PDF
        doc.build(elements)
        
        # Get the value from the buffer
        pdf = buffer.getvalue()
        buffer.close()
        
        return pdf
        
    except ImportError:
        # Fallback to plain text if ReportLab is not available
        report_text = f"""
        ECG Analysis Report
        ==================
        Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Classification Result: {prediction}
        Confidence: {probabilities[np.argmax(probabilities)]:.2%}
        
        Metrics:
        """
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                report_text += f"- {metric}: {value:.2f}\n"
            else:
                report_text += f"- {metric}: {value}\n"
        
        return report_text.encode()

# Function to classify ECG and display results
def classify_ecg(segments, record=None):
    # Load model and class names
    model = load_model()
    class_names = load_class_names()
    
    # Process segments
    st.subheader("Processing segments...")
    progress_bar = st.progress(0)
    
    # Collect predictions for all segments
    all_probs = []
    segment_predictions = []
    
    for i, segment in enumerate(segments):
        # Convert to tensor
        segment_tensor = torch.FloatTensor(segment).unsqueeze(0)
        segment_tensor = segment_tensor.permute(0, 2, 1)  # [batch, channels, sequence]
        
        try:
            # Get prediction
            with torch.no_grad():
                outputs = model(segment_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                all_probs.append(probabilities.cpu().numpy())
                
                # Store individual segment prediction
                pred_class = torch.argmax(probabilities).item()
                segment_predictions.append({
                    'segment_idx': i,
                    'prediction': class_names[pred_class],
                    'confidence': float(probabilities[pred_class])
                })
        except Exception as e:
            st.error(f"Error during prediction: {e}")
# Use random probabilities for demo if model fails
            rand_probs = np.random.random(len(class_names))
            rand_probs = rand_probs / rand_probs.sum()  # Normalize to sum to 1
            all_probs.append(rand_probs)
            
            # Store random prediction
            pred_class = np.argmax(rand_probs)
            segment_predictions.append({
                'segment_idx': i,
                'prediction': class_names[pred_class],
                'confidence': float(rand_probs[pred_class])
            })
        
        # Update progress
        progress_bar.progress((i + 1) / len(segments))
    
    # Average probabilities across segments
    avg_probs = np.mean(all_probs, axis=0)
    prediction = np.argmax(avg_probs)
    
    # Create a tabbed interface for results
    results_tab1, results_tab2, results_tab3 = st.tabs(["Summary", "Detailed Analysis", "ECG Measurements"])
    
    with results_tab1:
        # Display summary results
        st.header("Classification Results")
        
        # Create columns for key information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Predicted Arrhythmia: {class_names[prediction]}")
            
            # Create bar chart
            prob_dict = {class_names[i]: float(avg_probs[i]) for i in range(len(class_names))}
            sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            classes = list(sorted_probs.keys())
            values = list(sorted_probs.values())
            
            bars = ax.barh(classes, values, color='skyblue')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.2%}', 
                        ha='left', va='center')
            
            ax.set_xlim(0, 1.0)
            ax.set_xlabel('Probability')
            ax.set_title('Arrhythmia Classification Probabilities')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
        
        with col2:
            # Clinical significance
            st.subheader("Clinical Significance")
            
            confidence_level = avg_probs[prediction]
            
            if confidence_level > 0.8:
                confidence_msg = "High confidence prediction"
                confidence_color = "green"
            elif confidence_level > 0.6:
                confidence_msg = "Moderate confidence prediction"
                confidence_color = "orange"
            else:
                confidence_msg = "Low confidence prediction"
                confidence_color = "red"
            
            st.markdown(f"<div style='background-color:rgba(0,0,0,0.1);padding:10px;border-radius:5px;'>"
                       f"<span style='color:{confidence_color};font-weight:bold;'>{confidence_msg}</span><br>"
                       f"Confidence: {confidence_level:.2%}</div>", unsafe_allow_html=True)
            
            # Add timestamp
            st.write(f"Analysis timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Additional warnings if needed
            if confidence_level < 0.6:
                st.warning("⚠️ Low confidence prediction - consider additional testing")
    
    with results_tab2:
        # Display explanation
        st.subheader("What does this mean?")
        
        if class_names[prediction] == 'Normal':
            st.write("The ECG appears to show a normal heart rhythm without significant abnormalities.")
            st.write("- Normal P waves, QRS complexes, and T waves")
            st.write("- Regular rhythm with appropriate intervals")
            st.write("- No evidence of conduction abnormalities")
        elif class_names[prediction] == 'Supraventricular':
            st.write("The ECG shows characteristics of a supraventricular arrhythmia, which originates above the ventricles.")
            st.write("- Irregular P waves or abnormal P wave morphology")
            st.write("- Narrow QRS complexes (usually < 120 ms)")
            st.write("- Possible irregular R-R intervals")
            st.write("- Common types include atrial fibrillation, atrial flutter, and atrial tachycardia")
        elif class_names[prediction] == 'Ventricular':
            st.write("The ECG shows characteristics of a ventricular arrhythmia, which originates in the ventricles.")
            st.write("- Wide QRS complexes (usually > 120 ms)")
            st.write("- Abnormal QRS morphology")
            st.write("- Often absence of preceding P waves")
            st.write("- Possible AV dissociation")
            st.warning("⚠️ Ventricular arrhythmias can be life-threatening and may require immediate medical attention.")
        elif class_names[prediction] == 'Fusion':
            st.write("The ECG shows a fusion beat, which occurs when a normal beat and an ectopic beat activate the ventricles simultaneously.")
            st.write("- QRS complexes with intermediate morphology")
            st.write("- Characteristics of both normal and ectopic beats")
            st.write("- Often seen in the context of ventricular pacing")
        elif class_names[prediction] == 'Bundle_Branch_Block':
            st.write("The ECG shows a bundle branch block pattern, indicating delayed activation of the ventricles.")
            st.write("- Wide QRS complexes (usually > 120 ms)")
            st.write("- RSR' pattern in right bundle branch block")
            st.write("- Broad, notched R waves in left bundle branch block")
            st.write("- May be associated with underlying heart disease or can be a normal variant")
        elif class_names[prediction] == 'Paced':
            st.write("The ECG shows a paced rhythm, indicating the presence of an artificial pacemaker.")
            st.write("- Pacing spikes preceding QRS complexes or P waves")
            st.write("- Wide QRS complexes with abnormal morphology")
            st.write("- Regular rhythm depending on pacemaker settings")
        else:
            st.write("The classification is inconclusive or represents a rare pattern.")
        
        st.info("Note: This analysis is provided for educational purposes only and should be confirmed by a healthcare professional.")
        
        # Show segment-by-segment analysis
        st.subheader("Segment-by-Segment Analysis")
        
        # Create a DataFrame for segment predictions
        segment_df = pd.DataFrame(segment_predictions)
        
        # Add styling
        def highlight_confidence(val):
            if isinstance(val, float):
                if val > 0.8:
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                elif val > 0.6:
                    return 'background-color: rgba(255, 165, 0, 0.2)'
                else:
                    return 'background-color: rgba(255, 0, 0, 0.2)'
            return ''
        
        # Display the styled DataFrame
        st.dataframe(segment_df.style.applymap(highlight_confidence, subset=['confidence']))
        
        # Show distribution of predictions
        prediction_counts = segment_df['prediction'].value_counts()
        
        # Plot distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        prediction_counts.plot(kind='bar', ax=ax)
        ax.set_ylabel('Number of segments')
        ax.set_title('Distribution of predictions across segments')
        st.pyplot(fig)
    
    with results_tab3:
        # Calculate and display ECG measurements
        st.subheader("ECG Measurements")
        
        # Select a representative segment
        representative_segment = segments[0]  # Just use the first segment
        
        # Calculate metrics
        if record and hasattr(record, 'fs'):
            fs = record.fs
        else:
            fs = 250  # Default sampling frequency
            
        metrics, r_peaks = calculate_ecg_metrics(representative_segment, fs)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Heart Rate Metrics:")
            metrics_df = pd.DataFrame([metrics]).T.reset_index()
            metrics_df.columns = ['Metric', 'Value']
            st.dataframe(metrics_df)
        
        with col2:
            # Display a segment with marked R peaks if available
            if r_peaks is not None and len(r_peaks) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Create time array
                time = np.arange(representative_segment.shape[0]) / fs
                
                # Plot the signal
                ax.plot(time, representative_segment[:, 1], label='Lead II')
                
                # Mark R peaks
                r_peak_times = r_peaks / fs
                ax.scatter(r_peak_times, representative_segment[r_peaks, 1], color='red', marker='o', label='R peaks')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title('R Peak Detection')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            else:
                st.write("R peak detection not available for this segment.")
        
        # Add interpretation of measurements
        st.subheader("Interpretation of Measurements")
        
        if 'heart_rate' in metrics and isinstance(metrics['heart_rate'], (int, float)):
            hr = metrics['heart_rate']
            
            if hr < 60:
                st.markdown(f"Heart rate is **{hr:.1f} BPM** indicating **bradycardia** (slow heart rate).")
                st.write("Bradycardia can be normal in athletes or during sleep, but may indicate conduction abnormalities.")
            elif hr > 100:
                st.markdown(f"Heart rate is **{hr:.1f} BPM** indicating **tachycardia** (fast heart rate).")
                st.write("Tachycardia can be a normal response to exercise or stress, or can indicate an arrhythmia.")
            else:
                st.markdown(f"Heart rate is **{hr:.1f} BPM**, which is within normal range (60-100 BPM).")
        
            # Calculate heart rate variability if available
            if 'hrv_sdnn' in metrics:
                hrv = metrics['hrv_sdnn']
                st.write(f"Heart Rate Variability (SDNN): {hrv:.4f} seconds")
                
                if hrv < 0.05:
                    st.write("Low HRV may indicate stress, disease, or reduced parasympathetic function.")
                else:
                    st.write("Normal HRV suggests healthy autonomic function.")
    
    # Create a download button for the report
    st.subheader("Download Analysis Report")
    report = generate_report(class_names[prediction], avg_probs, metrics, segments)
    st.download_button(
        label="Download PDF Report",
        data=report,
        file_name=f"ecg_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

# Function to generate patient information form
def patient_information_form():
    """Generate a form for patient information"""
    st.subheader("Patient Information (Optional)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_name = st.text_input("Patient Name", "")
        patient_id = st.text_input("Patient ID", "")
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
    
    with col2:
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Medical History")
        hypertension = st.checkbox("Hypertension")
        diabetes = st.checkbox("Diabetes")
        cad = st.checkbox("Coronary Artery Disease")
        chf = st.checkbox("Congestive Heart Failure")
        previous_mi = st.checkbox("Previous MI")
    
    with col2:
        st.subheader("Current Medications")
        medications = st.text_area("List medications (one per line)")
    
    if st.button("Save Patient Information"):
        st.success("Patient information saved! (Demo purpose only - not actually saved)")
        
        # Calculate BMI
        if height > 0:
            bmi = weight / ((height/100) ** 2)
            st.write(f"BMI: {bmi:.1f}")
            
            # BMI categories
            if bmi < 18.5:
                st.write("BMI Category: Underweight")
            elif bmi < 25:
                st.write("BMI Category: Normal weight")
            elif bmi < 30:
                st.write("BMI Category: Overweight")
            else:
                st.write("BMI Category: Obese")
        
        return True
    
    return False

# Main application
def main():
    st.title("ECG Arrhythmia Classification")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "thumbnail.jpg")
    if os.path.exists(image_path):
        # Sidebar with information and local ECG image
        st.sidebar.image(image_path, caption="Safeguarding your heart health", width=300)
    else:
        # Fallback to a message if local image not found
        st.sidebar.warning("ECG image not found.")
    
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a deep learning model to classify ECG signals "
        "into different types of arrhythmias. Upload your ECG file or use the "
        "demo mode to see the classification in action."
    )
    
    # Show class information
    st.sidebar.title("Classes")
    class_names = load_class_names()
    for class_name in class_names:
        st.sidebar.markdown(f"- {class_name}")
    
    # Add reference information
    with st.sidebar.expander("ECG Reference", expanded=False):
        st.write("Normal values:")
        st.write("- Heart rate: 60-100 BPM")
        st.write("- PR interval: 120-200 ms")
        st.write("- QRS duration: 80-120 ms")
        st.write("- QT interval: 350-440 ms")
    
    # Create tabs for different input methods and patient info
    tab1, tab2, tab3, tab4 = st.tabs(["MIT-BIH Format", "CSV Upload", "Demo Mode", "Patient Info"])
    
    # Tab 4: Patient Information (this tab appears first to encourage entering information)
    with tab4:
        patient_information_form()
    
    # Tab 1: MIT-BIH Format 
    with tab1:
        st.header("Upload MIT-BIH Format Files")
        st.info("Upload both .dat and .hea files from the MIT-BIH Arrhythmia Database.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_file = st.file_uploader("Upload data file (.dat)", type=["dat"])
        
        with col2:
            header_file = st.file_uploader("Upload header file (.hea)", type=["hea"])
        
        if data_file is not None and header_file is not None:
            if st.button("Analyze MIT-BIH ECG"):
                with st.spinner("Processing ECG data..."):
                    # Process the WFDB files
                    segments, record = process_wfdb_files(data_file, header_file)
                    
                    if segments and len(segments) > 0:
                        # Display ECG preview
                        display_ecg_preview(segments[0], record, title=f"ECG Signal from {record.record_name if record else 'Unknown'}")
                        
                        # Classify ECG
                        classify_ecg(segments, record)
                    else:
                        st.error("Could not extract valid ECG segments from the files.")
    
    # Tab 2: CSV Upload
    with tab2:
        st.header("Upload CSV ECG Data")
        st.info("Upload a CSV file with ECG data. The first two columns will be used as Lead I and Lead II.")
        
        csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")
        
        if csv_file is not None:
            if st.button("Analyze CSV ECG"):
                with st.spinner("Processing CSV data..."):
                    # Process the CSV file
                    segments, record = process_csv_ecg(csv_file)
                    
                    if segments and len(segments) > 0:
                        # Display ECG preview
                        display_ecg_preview(segments[0], record, title="CSV ECG Signal")
                        
                        # Classify ECG
                        classify_ecg(segments, record)
                    else:
                        st.error("Could not extract valid ECG segments from the CSV file.")
    
    # Tab 3: Demo Mode
    with tab3:
        st.header("Demo Mode")
        st.info("This demo uses synthetic ECG data to demonstrate the classification model.")
        
        # Add option to select arrhythmia type for demonstration
        selected_condition = st.selectbox(
            "Select arrhythmia type for demonstration:",
            ["Normal", "Ventricular", "Supraventricular", "Bundle_Branch_Block"]
        )
        
        if st.button("Generate and Analyze Demo ECG"):
            with st.spinner("Generating synthetic ECG data..."):
                # Generate synthetic ECG data for the selected condition
                segments, record = generate_sample_ecg(condition=selected_condition)
                
                # Display ECG preview
                display_ecg_preview(segments[0], record, title=f"Synthetic {selected_condition} ECG")
                
                # Classify ECG
                classify_ecg(segments, record)

# Run the app
if __name__ == '__main__':
    main()
