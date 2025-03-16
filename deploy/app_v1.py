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

from model import XceptionTime

# Set up page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Classification",
    page_icon="❤️",
    layout="wide"
)

# Function to create a synthetic ECG image for the sidebar
def create_ecg_image():
    # Generate time points
    t = np.linspace(0, 4, 500)
    
    # Create synthetic ECG pattern (P wave, QRS complex, T wave)
    ecg = np.zeros_like(t)
    
    # Function to create repeating pattern
    def ecg_pattern(t_local):
        # P wave
        p_wave = 0.25 * np.sin(2*np.pi*t_local) * (t_local < 0.2)
        
        # QRS complex (sharper and higher amplitude)
        qrs = np.zeros_like(t_local)
        qrs_mask = (t_local >= 0.25) & (t_local < 0.35)
        qrs[qrs_mask] = -0.25 * np.sin(8*np.pi*(t_local[qrs_mask]-0.25))
        qrs_peak = (t_local >= 0.3) & (t_local < 0.32)
        qrs[qrs_peak] = 1.0
        
        # T wave
        t_wave = 0.35 * np.sin(2*np.pi*(t_local-0.45)) * ((t_local >= 0.45) & (t_local < 0.7))
        
        return p_wave + qrs + t_wave
    
    # Create multiple heartbeats
    for i in range(5):
        segment = t[(t >= i*0.8) & (t < (i+1)*0.8)]
        if len(segment) > 0:
            local_t = (segment - i*0.8) / 0.8
            ecg[(t >= i*0.8) & (t < (i+1)*0.8)] = ecg_pattern(local_t)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(t, ecg, color='#00AB66', linewidth=1.5)
    ax.axis('off')
    ax.set_ylim(-0.4, 1.2)
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

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
        model.load_state_dict(torch.load('xceptiontime_best.pth', map_location=torch.device('cpu')))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Running in demo mode with random predictions.")
    
    model.eval()
    return model

# Function to load the class names
def load_class_names():
    try:
        # Try to load from saved file first
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
        
        # Debug information
        #st.write(f"Files saved to: {temp_dir}")
        #st.write(f"Data file exists: {os.path.exists(data_path)}")
        #st.write(f"Header file exists: {os.path.exists(header_path)}")
        
        # List directory contents for reference
        #st.write("Directory contents:")
        #st.write(os.listdir(temp_dir))
        
        # Try to read the record
        record_path = os.path.join(temp_dir, original_record_name)
        try:
            record = wfdb.rdrecord(record_path)
            st.success("WFDB file successfully loaded!")
            
            # Convert to DataFrame for visualization
            signal_df = pd.DataFrame(record.p_signal, 
                                    columns=[f'Lead {i+1}' for i in range(record.p_signal.shape[1])])
            st.write("First few rows of signal data:")
            st.write(signal_df.head())
            
            # Extract signals
            signals = record.p_signal
            
            # Check if we have at least 2 channels
            if signals.shape[1] < 2:
                st.error(f"The ECG file has only {signals.shape[1]} channel(s). We need at least 2.")
                return None
            
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
            
            return segments
        
        except Exception as e:
            st.error(f"Error reading WFDB record: {e}")
            # List directory contents for debugging
            st.write("Directory contents:")
            st.write(os.listdir(temp_dir))
            
            # Check if we need to try a different approach
            try:
                # Try reading with wfdb.rdsamp which is sometimes more forgiving
                signals, fields = wfdb.rdsamp(record_path)
                st.success("Successfully read signals using rdsamp!")
                
                # Check if we have at least 2 channels
                if signals.shape[1] < 2:
                    st.error(f"The ECG file has only {signals.shape[1]} channel(s). We need at least 2.")
                    return None
                
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
                    return segments
                else:
                    st.error("Could not extract any segments from the ECG.")
                    return None
                
            except Exception as alt_e:
                st.error(f"Alternative method also failed: {alt_e}")
                return None
    
    except Exception as e:
        st.error(f"Error processing WFDB files: {e}")
        return None
    
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
            return None
        
        # Use the first two columns as Lead I and Lead II
        signals = df.iloc[:, :2].values
        
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
        
        return segments
    
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None

# Function to generate synthetic ECG data for demonstration
def generate_sample_ecg(window_size=256, num_segments=5):
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
    
    return segments

# Function to display ECG preview
def display_ecg_preview(segment):
    st.subheader("ECG Signal Preview")
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(segment.shape[0])
    ax.plot(time, segment[:, 0], label='Lead I')
    ax.plot(time, segment[:, 1], label='Lead II')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude (normalized)')
    ax.set_title('ECG Signal')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Function to classify ECG and display results
def classify_ecg(segments):
    # Load model and class names
    model = load_model()
    class_names = load_class_names()
    
    # Process segments
    st.subheader("Processing segments...")
    progress_bar = st.progress(0)
    
    # Collect predictions for all segments
    all_probs = []
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
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            # Use random probabilities for demo if model fails
            rand_probs = np.random.random(len(class_names))
            rand_probs = rand_probs / rand_probs.sum()  # Normalize to sum to 1
            all_probs.append(rand_probs)
        
        # Update progress
        progress_bar.progress((i + 1) / len(segments))
    
    # Average probabilities across segments
    avg_probs = np.mean(all_probs, axis=0)
    prediction = np.argmax(avg_probs)
    
    # Display results
    st.header("Classification Results")
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
    
    # Display explanation
    st.subheader("What does this mean?")
    
    if class_names[prediction] == 'Normal':
        st.write("The ECG appears to show a normal heart rhythm without significant abnormalities.")
    elif class_names[prediction] == 'Supraventricular':
        st.write("The ECG shows characteristics of a supraventricular arrhythmia, which originates above the ventricles.")
    elif class_names[prediction] == 'Ventricular':
        st.write("The ECG shows characteristics of a ventricular arrhythmia, which originates in the ventricles.")
    elif class_names[prediction] == 'Fusion':
        st.write("The ECG shows a fusion beat, which occurs when a normal beat and an ectopic beat activate the ventricles simultaneously.")
    elif class_names[prediction] == 'Bundle_Branch_Block':
        st.write("The ECG shows a bundle branch block pattern, indicating delayed activation of the ventricles.")
    elif class_names[prediction] == 'Paced':
        st.write("The ECG shows a paced rhythm, indicating the presence of an artificial pacemaker.")
    else:
        st.write("The classification is inconclusive or represents a rare pattern.")
    
    st.info("Note: This analysis is provided for educational purposes only and should be confirmed by a healthcare professional.")

# Main application
def main():
    st.title("ECG Arrhythmia Classification")
    
    # Sidebar with information and synthetic ECG image
    st.sidebar.image(create_ecg_image(), caption="Synthetic ECG", width=300)
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
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["MIT-BIH Format", "CSV Upload", "Demo Mode"])
    
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
                    segments = process_wfdb_files(data_file, header_file)
                    
                    if segments and len(segments) > 0:
                        # Display ECG preview
                        display_ecg_preview(segments[0])
                        
                        # Classify ECG
                        classify_ecg(segments)
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
                    segments = process_csv_ecg(csv_file)
                    
                    if segments and len(segments) > 0:
                        # Display ECG preview
                        display_ecg_preview(segments[0])
                        
                        # Classify ECG
                        classify_ecg(segments)
                    else:
                        st.error("Could not extract valid ECG segments from the CSV file.")
    
    # Tab 3: Demo Mode
    with tab3:
        st.header("Demo Mode")
        st.info("This demo uses synthetic ECG data to demonstrate the classification model.")
        
        if st.button("Generate and Analyze Demo ECG"):
            with st.spinner("Generating synthetic ECG data..."):
                # Generate synthetic ECG data
                segments = generate_sample_ecg()
                
                # Display ECG preview
                display_ecg_preview(segments[0])
                
                # Classify ECG
                classify_ecg(segments)

# Run the app
if __name__ == '__main__':
    main()