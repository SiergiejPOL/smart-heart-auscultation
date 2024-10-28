import numpy as np
import scipy.signal as signal
import pyaudio
import wave
import threading
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')

# Import reportlab for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch, mm

# Global variables
output_directory = None
countdown_label = None
fig, ax = None, None
canvas = None
line = None
buffer_size = 44100 * 10  # Buffer to store the last 10 seconds of data
data_buffer = np.zeros(buffer_size, dtype=np.float32)
chunk = 1024  # Record in chunks of 1024 samples
pvc_counter = 0  # Counter for PVCs (Premature Ventricular Contractions)
is_recording = False  # Flag to check if recording is ongoing
heart_icon_label = None  # Label to display the heart icon
heart_animation_running = False  # Flag to track the animation state
current_bpm = 0  # Current BPM value
animation_id = None  # Tkinter's after() ID for animation
recent_bpms = []  # List to store recent BPMs for smoothing
is_smart_auscultation_running = False
current_auscultation_step = 0  # Initialize auscultation step
patient_info = {}  # Dictionary to store patient information
record_buttons = []  # List to store recording buttons to enable/disable

# List to store recording information for SMART Auscultation
smart_recordings = []

# SMART Auscultation Points Information
auscultation_points = [
    # Aortic Valve
    ("Aortic Valve (Normal Breathing)",
     "Place the stethoscope in the second intercostal space on the right side of the sternum.\nBreathe normally for 60 seconds during recording.",
     "Aortic Valve - Normal", 60),
    ("Aortic Valve (Breathing Tasks)",
     "Place the stethoscope in the second intercostal space on the right side of the sternum.\nBreathe normally for 30 seconds, then take a deep breath, exhale slowly, and hold your breath for a few seconds.",
     "Aortic Valve - Task", 60),

    # Pulmonary Valve
    ("Pulmonary Valve (Normal Breathing)",
     "Move the stethoscope to the second intercostal space on the left side of the sternum.\nBreathe normally for 60 seconds during recording.",
     "Pulmonary Valve - Normal", 60),
    ("Pulmonary Valve (Breathing Tasks)",
     "Move the stethoscope to the second intercostal space on the left side of the sternum.\nTake a deep breath, hold it for 5-10 seconds, then breathe normally.",
     "Pulmonary Valve - Task", 60),

    # Erb's Point
    ("Erb's Point (Normal Breathing)",
     "Place the stethoscope in the third intercostal space on the left side of the sternum.\nBreathe normally for 60 seconds during recording.",
     "Erb's Point - Normal", 60),
    ("Erb's Point (Breathing Tasks)",
     "Place the stethoscope in the third intercostal space on the left side of the sternum.\nBreathe normally for 30 seconds, then take a deep breath, exhale slowly, and hold your breath.",
     "Erb's Point - Task", 60),

    # Tricuspid Valve
    ("Tricuspid Valve (Normal Breathing)",
     "Move the stethoscope to the fourth/fifth intercostal space on the left side of the sternum.\nBreathe normally for 60 seconds during recording.",
     "Tricuspid Valve - Normal", 60),
    ("Tricuspid Valve (Breathing Tasks)",
     "Move the stethoscope to the fourth/fifth intercostal space on the left side of the sternum.\nTake a deep breath, exhale slowly, and hold your breath for 5-10 seconds.",
     "Tricuspid Valve - Task", 60),

    # Mitral Valve
    ("Mitral Valve (Normal Breathing)",
     "Place the stethoscope in the fifth intercostal space on the left side, along the midclavicular line.\nBreathe normally for 60 seconds during recording.",
     "Mitral Valve - Normal", 60),
    ("Mitral Valve (Breathing Tasks)",
     "Place the stethoscope in the fifth intercostal space on the left side, along the midclavicular line.\nBreathe normally, then take a deep breath, exhale slowly, and hold your breath.",
     "Mitral Valve - Task", 60),
]

# Function to create the directory with date and time
def create_directory():
    global output_directory
    if output_directory is None:
        base_path = os.path.dirname(os.path.abspath(__file__))  # Default to script directory
        output_directory = filedialog.askdirectory(initialdir=base_path, title="Select Directory to Save Recordings")
        if not output_directory:
            messagebox.showwarning("No Directory Selected", "Please select a directory to save the recordings.")
            return False
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = os.path.join(output_directory, f"Heart Record {start_time}")
        os.makedirs(output_directory, exist_ok=True)
    return True

# Function to update the countdown
def update_countdown(time_left):
    if is_recording and time_left > 0:
        countdown_label.config(text=f"Recording ends in {time_left} seconds")
        root.after(1000, update_countdown, time_left - 1)
    else:
        countdown_label.config(text="Recording complete")

# Function to update the PCG plot
def update_plot():
    if is_recording:
        global line, data_buffer
        line.set_ydata(data_buffer)
        canvas.draw()

# Function to detect PVCs (extrasystoles)
def detect_pvc(intervals):
    pvc_indices = []
    mean_interval = np.mean(intervals)

    for i in range(1, len(intervals)):
        if intervals[i-1] < 0.5 * mean_interval and intervals[i] > 1.5 * mean_interval:
            pvc_indices.append(i)
        
    return pvc_indices

# Function to classify heart rhythm
def classify_heart_rhythm(heart_rate, intervals):
    if heart_rate < 60:
        rhythm = "Bradycardia"
    elif heart_rate > 100:
        rhythm = "Tachycardia"
    else:
        rhythm = "Normal Rhythm"

    # Check regularity
    if np.std(intervals) > 0.1 * np.mean(intervals):
        regularity = "Irregular Rhythm"
    else:
        regularity = "Regular Rhythm"
    
    return rhythm, regularity

# Bandpass Filter Function
def bandpass_filter(data, fs, lowcut=20.0, highcut=150.0, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    y = signal.sosfilt(sos, data)
    return y

# Function to calculate heart rate and identify PVCs
def calculate_heart_rate(data, fs):
    if np.max(np.abs(data)) == 0:
        return 0, [], []
    # Apply bandpass filter
    filtered_data = bandpass_filter(data, fs)
    # Compute envelope using Hilbert transform
    analytic_signal = signal.hilbert(filtered_data)
    envelope = np.abs(analytic_signal)
    normalized_data = envelope / np.max(envelope)
    # Adaptive threshold
    threshold = np.mean(normalized_data) * 1.2
    # Peak detection
    peaks, _ = find_peaks(normalized_data, distance=fs*0.3, height=threshold)
    if len(peaks) < 2:
        return 0, [], []
    intervals = np.diff(peaks) / fs
    pvc_indices = detect_pvc(intervals)
    valid_intervals = np.delete(intervals, pvc_indices)
    if len(valid_intervals) > 0:
        heart_rate = 60 / np.mean(valid_intervals)
    else:
        heart_rate = 0
    return heart_rate, intervals, pvc_indices

# Function to calculate HRV metrics
def calculate_hrv(intervals):
    if len(intervals) < 2:
        return {'SDNN': 'N/A', 'RMSSD': 'N/A'}

    rr_intervals = intervals * 1000  # Convert to milliseconds
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    return {'SDNN': round(sdnn, 2), 'RMSSD': round(rmssd, 2)}

# Function to animate the beating heart symbol (change color to simulate beat)
def animate_heart():
    global animation_id, current_bpm, heart_animation_running
    if not heart_animation_running or current_bpm == 0:
        return  # Do not animate if not running or BPM is 0

    heart_icon_label.config(foreground="red")
    
    try:
        interval = int(60000 / current_bpm / 2)
    except ZeroDivisionError:
        interval = 500  # Default interval if BPM is 0

    animation_id = root.after(interval, unbeat)

def unbeat():
    global animation_id, current_bpm, heart_animation_running
    if not heart_animation_running or current_bpm == 0:
        return  # Do not animate if not running or BPM is 0

    heart_icon_label.config(foreground="black")
    
    try:
        interval = int(60000 / current_bpm / 2)
    except ZeroDivisionError:
        interval = 500  # Default interval if BPM is 0
    
    animation_id = root.after(interval, animate_heart)

def start_heart_animation():
    global animation_id, heart_animation_running
    if not heart_animation_running:
        heart_animation_running = True
        animate_heart()

def stop_heart_animation():
    global animation_id, heart_animation_running
    heart_animation_running = False
    if animation_id is not None:
        root.after_cancel(animation_id)
        animation_id = None
    heart_icon_label.config(foreground="black")

# Function to generate PDF report
def generate_pdf_report(filename, valve_name, avg_heart_rate, recording_duration, waveform_image_paths, pvc_count, rhythm, regularity, hrv_metrics, heart_rate_trend_path):
    try:
        pdf_filename = os.path.join(output_directory, f"{valve_name}_Report.pdf")
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterTitle', alignment=TA_CENTER, fontSize=18, leading=22))
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

        # Patient Information
        elements.append(Paragraph("Patient Information", styles['Heading1']))
        patient_data = [
            ["Name:", patient_info.get('Name', 'N/A')],
            ["Age:", patient_info.get('Age', 'N/A')],
            ["Sex:", patient_info.get('Sex', 'N/A')],
            ["Height:", f"{patient_info.get('Height', 'N/A')} cm"],
            ["Weight:", f"{patient_info.get('Weight', 'N/A')} kg"],
            ["Medical History:", patient_info.get('Medical History', 'N/A')],
        ]
        table = Table(patient_data, colWidths=[100, 400])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Recording Summary
        elements.append(Paragraph("Recording Summary", styles['Heading1']))
        summary_data = [
            ["Valve", valve_name],
            ["Recording Duration", f"{recording_duration} seconds"],
            ["Average Heart Rate", f"{avg_heart_rate} BPM"],
            ["PVC Count", pvc_count],
            ["Rhythm", rhythm],
            ["Regularity", regularity],
            ["SDNN (ms)", hrv_metrics['SDNN']],
            ["RMSSD (ms)", hrv_metrics['RMSSD']]
        ]
        table = Table(summary_data, colWidths=[200, 300])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Heart Rate Trend Graph
        if heart_rate_trend_path:
            elements.append(Paragraph("Heart Rate Trend", styles['Heading2']))
            elements.append(Image(heart_rate_trend_path, width=6*inch, height=3*inch))
            elements.append(Spacer(1, 12))

        # Detailed Analysis
        elements.append(Paragraph("Detailed Analysis", styles['Heading1']))

        for idx, image_path in enumerate(waveform_image_paths):
            elements.append(Paragraph(f"PCG Segment {idx + 1}", styles['Heading2']))
            elements.append(Image(image_path, width=6*inch, height=3*inch))
            elements.append(Spacer(1, 12))

        # Footer with page numbers
        def add_page_number(canvas, doc):
            page_num = canvas.getPageNumber()
            text = f"Page {page_num}"
            canvas.drawRightString(200 * mm, 15 * mm, text)

        doc.build(elements, onLaterPages=add_page_number)
        print(f"PDF report saved as {pdf_filename}")
    except Exception as e:
        print(f"Error generating PDF report: {e}")

# Function to generate PDF report for SMART Auscultation
def generate_smart_pdf_report():
    try:
        pdf_filename = os.path.join(output_directory, "SMART_Auscultation_Report.pdf")
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterTitle', alignment=TA_CENTER, fontSize=18, leading=22))
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

        # Patient Information
        elements.append(Paragraph("Patient Information", styles['Heading1']))
        patient_data = [
            ["Name:", patient_info.get('Name', 'N/A')],
            ["Age:", patient_info.get('Age', 'N/A')],
            ["Sex:", patient_info.get('Sex', 'N/A')],
            ["Height:", f"{patient_info.get('Height', 'N/A')} cm"],
            ["Weight:", f"{patient_info.get('Weight', 'N/A')} kg"],
            ["Medical History:", patient_info.get('Medical History', 'N/A')],
        ]
        table = Table(patient_data, colWidths=[100, 400])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Recording Summary
        elements.append(Paragraph("Recording Summary", styles['Heading1']))
        summary_table_data = [["Valve", "Avg HR (BPM)", "PVC Count", "Rhythm", "Regularity", "SDNN (ms)", "RMSSD (ms)"]]
        for recording in smart_recordings:
            summary_table_data.append([
                recording['valve_name'],
                recording['avg_heart_rate'],
                recording['pvc_count'],
                recording['rhythm'],
                recording['regularity'],
                recording['hrv_metrics']['SDNN'],
                recording['hrv_metrics']['RMSSD']
            ])
        table = Table(summary_table_data, colWidths=[100, 60, 60, 80, 80, 60, 60])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER')
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Detailed Analysis
        elements.append(Paragraph("Detailed Analysis", styles['Heading1']))

        for recording in smart_recordings:
            elements.append(Paragraph(f"{recording['valve_name']}", styles['Heading2']))
            details_data = [
                ["Recording Duration", f"{recording['duration']} seconds"],
                ["Average Heart Rate", f"{recording['avg_heart_rate']} BPM"],
                ["PVC Count", recording['pvc_count']],
                ["Rhythm", recording['rhythm']],
                ["Regularity", recording['regularity']],
                ["SDNN (ms)", recording['hrv_metrics']['SDNN']],
                ["RMSSD (ms)", recording['hrv_metrics']['RMSSD']]
            ]
            table = Table(details_data, colWidths=[150, 350])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

            # Heart Rate Trend
            if recording['heart_rate_trend_path']:
                elements.append(Paragraph("Heart Rate Trend", styles['Heading3']))
                elements.append(Image(recording['heart_rate_trend_path'], width=6*inch, height=3*inch))
                elements.append(Spacer(1, 12))

            # PCG Plots
            for idx, image_path in enumerate(recording['waveform_images']):
                elements.append(Paragraph(f"PCG Segment {idx + 1}", styles['Heading3']))
                elements.append(Image(image_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 12))

        # Footer with page numbers
        def add_page_number(canvas, doc):
            page_num = canvas.getPageNumber()
            text = f"Page {page_num}"
            canvas.drawRightString(200 * mm, 15 * mm, text)

        doc.build(elements, onLaterPages=add_page_number)
        print(f"SMART Auscultation PDF report saved as {pdf_filename}")
    except Exception as e:
        print(f"Error generating SMART PDF report: {e}")

# Function to stop the recording and stop SMART Auscultation
def stop_recording():
    global is_recording, is_smart_auscultation_running, animation_id, heart_animation_running
    is_recording = False  # Stop recording

    # Stop the heart animation if running
    if heart_animation_running:
        stop_heart_animation()

    # Cancel any scheduled after() functions
    if animation_id is not None:
        root.after_cancel(animation_id)
        animation_id = None

    # Reset all labels
    status_label.config(text="Recording stopped.")
    countdown_label.config(text="")
    pvc_counter_label.config(text="PVC Counter: 0")
    heart_rate_label.config(text="Heart Rate: N/A")
    average_heart_rate_label.config(text="Average Heart Rate: N/A")
    rhythm_label.config(text="Rhythm: N/A")
    regularity_label.config(text="Regularity: N/A")
    heart_icon_label.config(foreground="black")  # Reset heart icon

    # Stop the SMART Auscultation process
    if is_smart_auscultation_running:
        stop_smart_auscultation()

def stop_smart_auscultation():
    global is_smart_auscultation_running, current_auscultation_step, animation_id, smart_recordings
    is_smart_auscultation_running = False  # Stop any ongoing auscultation
    current_auscultation_step = 0  # Reset the auscultation step
    info_label.config(text="SMART Auscultation stopped.")
    smart_recordings.clear()  # Clear any stored recordings
    
    # Cancel any scheduled after() functions
    if animation_id is not None:
        root.after_cancel(animation_id)
        animation_id = None

# Function to record audio and update the plot and heart rate in real-time
def record_audio(duration, filename, valve_name=None, is_smart=False):
    global pvc_counter, is_recording, current_bpm, recent_bpms
    is_recording = True
    recent_bpms.clear()  # Clear previous BPMs for new recording
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames
    heart_rates = []  # Store heart rates for averaging
    heart_rate_history = []  # For smoothing
    previous_heart_rate = None
    total_pvc_count = 0  # Total PVC count during recording
    rhythms = []
    regularities = []
    intervals_list = []  # Store all intervals for HRV calculation
    time_stamps = []  # Time stamps for heart rate trend

    update_countdown(duration)

    for i in range(0, int(fs / chunk * duration)):
        if not is_recording:
            break  # Stop recording if the flag is set to False

        try:
            data = stream.read(chunk, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading audio stream: {e}")
            break
        frames.append(np.frombuffer(data, dtype=np.int16).astype(np.float32))  # Store as numpy array

        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if len(audio_data) > 0:
            audio_data /= 32768.0  # Normalize to range -1.0 to 1.0
            if len(audio_data) > buffer_size:
                audio_data = audio_data[-buffer_size:]
            data_buffer[:-len(audio_data)] = data_buffer[len(audio_data):]
            data_buffer[-len(audio_data):] = audio_data

        if i % (44100 // (chunk * 10)) == 0:  # Update approximately every 100ms
            update_plot()

        if i % (fs // chunk) == 0 and i > 0:
            if len(frames) >= int(fs/chunk):
                data_to_analyze = np.concatenate(frames[-int(fs/chunk):])
                heart_rate, intervals, pvc_indices = calculate_heart_rate(data_to_analyze, fs)

                if previous_heart_rate is not None and abs(heart_rate - previous_heart_rate) > 20:
                    continue  # Ignore this heart rate measurement
                else:
                    heart_rates.append(heart_rate)
                    previous_heart_rate = heart_rate
                    time_stamps.append(i * chunk / fs)

                pvc_counter += len(pvc_indices)
                total_pvc_count += len(pvc_indices)
                pvc_counter_label.config(text=f"PVC Counter: {pvc_counter}")

                # Smoothing heart rate measurements
                heart_rate_history.append(heart_rate)
                window_size = 5
                if len(heart_rate_history) >= window_size:
                    smoothed_bpm = np.mean(heart_rate_history[-window_size:])
                else:
                    smoothed_bpm = np.mean(heart_rate_history)
                current_bpm = round(smoothed_bpm)

                avg_heart_rate = round(np.mean(heart_rates))

                heart_rate_label.config(text=f"Heart Rate: {current_bpm} BPM")
                average_heart_rate_label.config(text=f"Average Heart Rate: {avg_heart_rate} BPM")

                rhythm, regularity = classify_heart_rhythm(current_bpm, intervals)
                rhythms.append(rhythm)
                regularities.append(regularity)
                intervals_list.extend(intervals)
                rhythm_label.config(text=f"Rhythm: {rhythm}")
                regularity_label.config(text=f"Regularity: {regularity}")

                if current_bpm > 0:
                    start_heart_animation()

    stream.stop_stream()
    stream.close()
    p.terminate()

    if is_recording:
        # Ensure the output directory exists
        if not output_directory:
            if not create_directory():
                messagebox.showwarning("Directory Error", "Unable to create output directory.")
                is_recording = False
                return

        filepath = os.path.join(output_directory, filename)
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        # Save the entire recording
        wf.writeframes(b''.join([np.int16(frame).tobytes() for frame in frames]))
        wf.close()

        # Generate the PCG plots from the recording divided into 10-second segments
        full_data = np.concatenate(frames)
        total_samples = len(full_data)
        samples_per_segment = fs * 10  # 10 seconds per segment
        num_segments = int(np.ceil(total_samples / samples_per_segment))

        waveform_image_paths = []

        for segment_idx in range(num_segments):
            start_sample = segment_idx * samples_per_segment
            end_sample = min((segment_idx + 1) * samples_per_segment, total_samples)
            segment_data = full_data[start_sample:end_sample]
            time_axis_segment = np.linspace(start_sample / fs, end_sample / fs, num=end_sample - start_sample)

            plt.figure(figsize=(12, 6))
            plt.plot(time_axis_segment, segment_data / np.max(np.abs(segment_data)))
            plt.title(f"Phonocardiogram (PCG) - {valve_name} (Segment {segment_idx + 1})")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Normalized Amplitude")
            waveform_image_path = os.path.join(output_directory, f"{valve_name}_Waveform_Segment_{segment_idx + 1}.png")
            plt.savefig(waveform_image_path)
            plt.close()
            waveform_image_paths.append(waveform_image_path)

        # Generate Heart Rate Trend Graph
        if heart_rates and time_stamps:
            plt.figure(figsize=(8, 4))
            plt.plot(time_stamps, heart_rates, marker='o')
            plt.title("Heart Rate Trend")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Heart Rate (BPM)")
            heart_rate_trend_path = os.path.join(output_directory, f"{valve_name}_HeartRateTrend.png")
            plt.savefig(heart_rate_trend_path)
            plt.close()
        else:
            heart_rate_trend_path = None

        # Calculate HRV metrics
        hrv_metrics = calculate_hrv(np.array(intervals_list))

        # Calculate final average heart rate, rhythm, and regularity
        final_avg_heart_rate = round(np.mean(heart_rates)) if heart_rates else 0
        most_common_rhythm = max(set(rhythms), key=rhythms.count) if rhythms else "N/A"
        most_common_regularity = max(set(regularities), key=regularities.count) if regularities else "N/A"

        # Generate PDF report
        if not is_smart:
            generate_pdf_report(filename, valve_name, final_avg_heart_rate, duration, waveform_image_paths,
                                total_pvc_count, most_common_rhythm, most_common_regularity, hrv_metrics, heart_rate_trend_path)
        else:
            # Store recording information for SMART Auscultation report
            smart_recordings.append({
                'valve_name': valve_name,
                'avg_heart_rate': final_avg_heart_rate,
                'duration': duration,
                'waveform_images': waveform_image_paths,
                'pvc_count': total_pvc_count,
                'rhythm': most_common_rhythm,
                'regularity': most_common_regularity,
                'hrv_metrics': hrv_metrics,
                'heart_rate_trend_path': heart_rate_trend_path
            })

        status_label.config(text=f"Finished recording {filename}")
        average_heart_rate_label.config(text=f"Average Heart Rate: {final_avg_heart_rate} BPM")

        stop_heart_animation()

    is_recording = False

# Function to start the recording
def start_recording(valve_name, duration_var):
    global pvc_counter, is_recording, current_bpm, recent_bpms
    if is_recording:
        return  # Prevent starting a new recording if one is already in progress

    pvc_counter = 0  # Reset PVC counter at the start of recording
    recent_bpms.clear()  # Clear previous BPMs for new recording
    
    if isinstance(duration_var, int):
        duration = duration_var
    else:
        duration = int(duration_var.get())
    
    filename = f"{valve_name}.wav"
    status_label.config(text=f"Recording {valve_name}...")
    heart_rate_label.config(text="Heart Rate: N/A")
    average_heart_rate_label.config(text="Average Heart Rate: N/A")
    pvc_counter_label.config(text="PVC Counter: 0")
    rhythm_label.config(text="Rhythm: N/A")
    regularity_label.config(text="Regularity: N/A")
    threading.Thread(target=record_audio, args=(duration, filename, valve_name, False), daemon=True).start()

# SMART Auscultation functions
def start_smart_auscultation():
    global is_smart_auscultation_running
    if not create_directory():
        return
    is_smart_auscultation_running = True  # Mark that the process is running
    start_welcome_message()

def start_welcome_message():
    messagebox.showinfo("Welcome", "SMART Auscultation guides you through a full heart auscultation. Sit comfortably and ensure the stethoscope is working properly.")
    root.after(500, test_stethoscope)  # Reduced delay to see if it triggers correctly

def test_stethoscope():
    update_info("Starting a 15-second stethoscope test to check the heartbeat.\nEnsure the stethoscope is placed correctly.")
    messagebox.showinfo("Stethoscope Test", "Starting a 15-second stethoscope test to check the heartbeat.")
    start_recording_with_callback("Stethoscope Test", 15, lambda: auscultation_process(0))

def start_recording_with_callback(valve_name, duration, callback):
    global pvc_counter, is_recording, current_bpm, recent_bpms
    if is_recording:
        return  # Prevent starting a new recording if one is already in progress

    pvc_counter = 0  # Reset PVC counter at the start of recording
    recent_bpms.clear()  # Clear previous BPMs for new recording

    # Define the filename for the recording
    filename = f"{valve_name}.wav"
    status_label.config(text=f"Recording {valve_name}...")
    heart_rate_label.config(text="Heart Rate: N/A")
    average_heart_rate_label.config(text="Average Heart Rate: N/A")
    pvc_counter_label.config(text="PVC Counter: 0")
    rhythm_label.config(text="Rhythm: N/A")
    regularity_label.config(text="Regularity: N/A")

    # Internal function to handle recording in a thread
    def recording_thread():
        record_audio(duration, filename, valve_name, is_smart=True)
        if callback:
            callback()

    # Start recording in a new thread
    threading.Thread(target=recording_thread, daemon=True).start()

def display_auscultation_info(index):
    """Update the info_label to show step-by-step instructions for each valve."""
    if index < len(auscultation_points):
        title, instructions, _, _ = auscultation_points[index]
        info_label.config(text=f"Point {index + 1}: {title}\n{instructions}")

def auscultation_process(index):
    global is_smart_auscultation_running, current_auscultation_step

    if not is_smart_auscultation_running:
        return  # Exit if auscultation has been stopped

    if index >= len(auscultation_points):
        end_smart_auscultation()
        return

    current_auscultation_step = index  # Update the current step
    display_auscultation_info(index)
    
    title, message, filename, duration = auscultation_points[index]
    messagebox.showinfo(f"Point {index + 1}: {title}", message)
    
    start_recording_with_callback(filename, duration, lambda: auscultation_process(index + 1))

# Function to update the info_label
def update_info(message):
    """Update the info_label with a given message."""
    info_label.config(text=message)

def end_smart_auscultation():
    generate_smart_pdf_report()
    messagebox.showinfo("End of SMART Auscultation", "Auscultation completed. You can review the recordings.")
    info_label.config(text="SMART Auscultation completed.")

# Function to check if all patient fields are filled
def check_patient_info():
    for key, var in patient_entries.items():
        if key == "Medical History":
            content = var.get("1.0", "end-1c").strip()
            if not content:
                return False
        elif key == "Sex":
            if not var.get():
                return False
        else:
            if not var.get().strip():
                return False
    return True

# Function to save patient data
def save_patient_data():
    global patient_info
    if not check_patient_info():
        messagebox.showwarning("Incomplete Information", "Please fill in all patient information fields.")
        return

    # Save patient information
    patient_info = {}
    for key, var in patient_entries.items():
        if key == "Medical History":
            content = var.get("1.0", "end-1c").strip()
            patient_info[key] = content
        else:
            patient_info[key] = var.get().strip()
    messagebox.showinfo("Patient Data Saved", "Patient information has been saved successfully.")

    # Enable recording buttons
    for btn in record_buttons:
        btn.config(state="normal")
    smart_button.config(state="normal")

    # Prompt for directory selection
    if not create_directory():
        # If directory selection was canceled, disable the buttons again
        for btn in record_buttons:
            btn.config(state="disabled")
        smart_button.config(state="disabled")
        return

# GUI setup
root = tk.Tk()
root.title("Heart Valve Sound Recorder & SMART Auscultation")
root.geometry("1480x820")
root.configure(bg="white")  # Set the background to white for the main window

# Setup style for frames and labels
style = ttk.Style()
style.configure("White.TFrame", background="white")
style.configure("White.TLabel", background="white", font=("Calibri", 16))
style.configure("White.TSeparator", background="white")
style.configure("TButton", font=("Calibri", 12))
style.configure("White.TEntry", fieldbackground="white", background="white")  # Set background color for Entry widgets

# Create frames for GUI components
top_frame = ttk.Frame(root, style="White.TFrame")
top_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

# Patient Information Frame
patient_frame = ttk.Frame(top_frame, style="White.TFrame", relief="groove")
patient_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
patient_frame.grid_columnconfigure(1, weight=1)

ttk.Label(patient_frame, text="Patient Information", style="White.TLabel", font=("Calibri", 18, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

patient_fields = ["Name", "Age", "Sex", "Height", "Weight", "Medical History"]
patient_entries = {}
for i, field in enumerate(patient_fields):
    ttk.Label(patient_frame, text=f"{field}:", style="White.TLabel").grid(row=i+1, column=0, sticky="e", padx=5, pady=2)
    if field == "Medical History":
        entry_widget = tk.Text(patient_frame, width=30, height=5, bg="white")
        entry_widget.grid(row=i+1, column=1, padx=5, pady=2, sticky="w")
        patient_entries[field] = entry_widget
    elif field == "Sex":
        entry_var = tk.StringVar()
        sex_options = ["Male", "Female"]
        entry_widget = ttk.Combobox(patient_frame, textvariable=entry_var, values=sex_options, state="readonly", width=28)
        entry_widget.grid(row=i+1, column=1, padx=5, pady=2, sticky="w")
        patient_entries[field] = entry_var
    elif field == "Height":
        entry_var = tk.StringVar()
        height_frame = ttk.Frame(patient_frame)
        height_frame.grid(row=i+1, column=1, padx=5, pady=2, sticky="w")
        entry_widget = tk.Entry(height_frame, textvariable=entry_var, width=25, bg="white")
        entry_widget.pack(side="left")
        unit_label = ttk.Label(height_frame, text="cm", style="White.TLabel")
        unit_label.pack(side="left", padx=5)
        patient_entries[field] = entry_var
    elif field == "Weight":
        entry_var = tk.StringVar()
        weight_frame = ttk.Frame(patient_frame)
        weight_frame.grid(row=i+1, column=1, padx=5, pady=2, sticky="w")
        entry_widget = tk.Entry(weight_frame, textvariable=entry_var, width=25, bg="white")
        entry_widget.pack(side="left")
        unit_label = ttk.Label(weight_frame, text="kg", style="White.TLabel")
        unit_label.pack(side="left", padx=5)
        patient_entries[field] = entry_var
    else:
        entry_var = tk.StringVar()
        ttk.Entry(patient_frame, textvariable=entry_var, width=30, style="White.TEntry").grid(row=i+1, column=1, padx=5, pady=2, sticky="w")
        patient_entries[field] = entry_var

# Save Patient Data Button
save_patient_button = ttk.Button(patient_frame, text="Save Patient Data", command=save_patient_data)
save_patient_button.grid(row=len(patient_fields)+1, column=0, columnspan=2, pady=10)

# Valve Recording Frame
valve_frame = ttk.Frame(top_frame, style="White.TFrame", relief="groove")
valve_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
valve_frame.grid_columnconfigure(1, weight=1)

ttk.Label(valve_frame, text="Valve Recordings", style="White.TLabel", font=("Calibri", 18, "bold")).grid(row=0, column=0, columnspan=3, pady=5)

valve_names = ["Aortic Valve", "Pulmonary Valve", "Erb's Point", "Tricuspid Valve", "Mitral Valve"]
duration_vars = {}

# Create duration inputs and record buttons for each valve
for i, valve in enumerate(valve_names):
    ttk.Label(valve_frame, text=f"{valve} duration (seconds):", style="White.TLabel").grid(row=i+1, column=0, padx=5, pady=5, sticky="e")
    duration_vars[valve] = tk.StringVar(value="60")
    ttk.Entry(valve_frame, textvariable=duration_vars[valve], width=10, style="White.TEntry").grid(row=i+1, column=1, padx=5, pady=5)
    btn = ttk.Button(valve_frame, text=f"Record {valve}", command=lambda v=valve, dv=duration_vars[valve]: start_recording(v, dv), state="disabled")
    btn.grid(row=i+1, column=2, padx=5, pady=5)
    record_buttons.append(btn)

# Button to stop recording
stop_button = ttk.Button(valve_frame, text="Stop Recording", command=stop_recording)
stop_button.grid(row=len(valve_names)+1, column=2, padx=5, pady=5)

# Button to start SMART Auscultation
smart_button = ttk.Button(valve_frame, text="Start SMART Auscultation", command=start_smart_auscultation, state="disabled")
smart_button.grid(row=len(valve_names)+2, column=0, columnspan=3, padx=10, pady=20)

# Information Frame
info_frame = ttk.Frame(top_frame, style="White.TFrame", relief="groove")
info_frame.grid(row=0, column=2, padx=15, pady=10, sticky="nsew")
info_frame.grid_columnconfigure(0, weight=1)

# Label for displaying dynamic information
info_label = ttk.Label(info_frame, text="SMART Heart Auscultation Information", style="White.TLabel", wraplength=400, font=("Calibri", 16))
info_label.grid(row=0, column=0, padx=10, pady=10)

# Phonocardiogram (PCG) plot
left_frame = ttk.Frame(root)
left_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
root.grid_columnconfigure(0, weight=5)  # 5/8 for PCG plot

fig, ax = plt.subplots(figsize=(8, 4))  # Reduced width, 5/8 proportions
ax.set_facecolor("white")
ax.set_ylim([-1.0, 1.0])
ax.set_xlim([0, 10])  # Display 10 seconds of time
ax.set_ylabel("Amplitude", fontsize=14)
ax.set_xlabel("Time (seconds)", fontsize=14)
ax.set_title("Phonocardiograph (PCG)", fontsize=16, pad=20)  # Set the title of the plot
time_axis = np.linspace(0, 10, buffer_size)  # Time axis from 0 to 10 seconds
line, = ax.plot(time_axis, np.zeros(buffer_size), lw=1.5)

# Adjust layout to fill empty spaces
fig.tight_layout(pad=0)
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.15)

canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Separator between PCG and status information section
vertical_separator = ttk.Separator(root, orient='vertical')
vertical_separator.grid(row=2, column=1, rowspan=1, sticky="ns", pady=10)

# Create a dedicated right frame for status information
right_frame = ttk.Frame(root, style="White.TFrame")
right_frame.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")
root.grid_columnconfigure(2, weight=3)  # 3/8 for information

# Status labels
status_label = ttk.Label(right_frame, text="Waiting to start recording...", style="White.TLabel")
status_label.grid(row=0, column=0, padx=10, pady=10)

countdown_label = ttk.Label(right_frame, text="", style="White.TLabel")
countdown_label.grid(row=1, column=0, padx=10, pady=10)

pvc_counter_label = ttk.Label(right_frame, text="PVC Counter: 0", style="White.TLabel")
pvc_counter_label.grid(row=4, column=0, padx=0, pady=0)

heart_rate_label = ttk.Label(right_frame, text="Heart Rate: N/A", style="White.TLabel")
heart_rate_label.grid(row=5, column=1, padx=10, pady=10)

average_heart_rate_label = ttk.Label(right_frame, text="Average Heart Rate: N/A", style="White.TLabel")
average_heart_rate_label.grid(row=5, column=0, padx=10, pady=10)

# Heart icon
heart_icon_label = ttk.Label(right_frame, text="♥", font=("Calibri", 116), foreground="black", background="white")
heart_icon_label.grid(row=4, column=1, padx=10, pady=0, sticky="w")

rhythm_label = ttk.Label(right_frame, text="Rhythm: N/A", style="White.TLabel")
rhythm_label.grid(row=6, column=1, padx=0, pady=0, sticky="w")

regularity_label = ttk.Label(right_frame, text="Regularity: N/A", style="White.TLabel")
regularity_label.grid(row=6, column=0, padx=0, pady=0)

# Start the Tkinter event loop
root.mainloop()
