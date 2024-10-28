 Heart Valve Sound Recorder & SMART Auscultation
 
![DALL·E 2024-10-27 19 52 57 - A heart-shaped icon with a stethoscope draped around it, a blood pressure monitor, and an EKG rhythm line displayed prominently in the center  The bac](https://github.com/user-attachments/assets/b2960bea-94d7-41a3-b009-c50a3ea9b43c)

This application is a comprehensive tool designed to record heart sounds from different heart valves, perform real-time analysis, detect potential anomalies like Premature Ventricular Contractions (PVCs), and generate detailed PDF reports. It also includes a SMART Auscultation feature that guides users through a full heart auscultation process with step-by-step instructions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Running the Application](#running-the-application)
  - [Patient Information](#patient-information)
  - [Recording Heart Sounds](#recording-heart-sounds)
  - [SMART Auscultation](#smart-auscultation)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Heart Sound Recording**: Record heart sounds from different valves with customizable durations.
- **Real-time Visualization**: Display real-time Phonocardiogram (PCG) plots during recording.
- **Heart Rate Calculation**: Calculate and display the current and average heart rate in BPM.
- **PVC Detection**: Detect Premature Ventricular Contractions using an improved algorithm based on median and MAD.
- **HRV Metrics**: Calculate Heart Rate Variability metrics like SDNN and RMSSD.
- **Rhythm Classification**: Classify heart rhythm as Bradycardia, Tachycardia, or Normal.
- **Regularity Assessment**: Determine if the rhythm is regular or irregular.
- **Animated Heart Icon**: Visual representation of the heartbeat synchronized with the calculated BPM.
- **PDF Report Generation**: Generate detailed PDF reports with patient information, recordings summary, PCG plots, heart rate trends, and RR intervals.
- **SMART Auscultation**: Guided auscultation process with step-by-step instructions for different heart valves and conditions.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/SiergiejPOL/smart-heart-auscultation.git
   cd heart-valve-sound-recorder
   
2. **Install Dependencies**

Dependencies:
- Python 3.x
- numpy
- scipy
- pyaudio
- wave
- tkinter
- matplotlib
- reportlab

Note: For Windows users, you may need to install pyaudio using a precompiled binary due to compilation issues.

3. **Patient Information**
Fill in Patient Details: Enter the patient's name, age, sex, height, weight, and medical history.
Save Patient Data: Click the "Save Patient Data" button. This information will be included in the generated PDF reports.
Select Output Directory: Upon saving, you'll be prompted to select a directory to save the recordings and reports.

4. **Recording Heart Sounds**
Set Recording Duration: For each heart valve, you can set a custom recording duration in seconds.
Start Recording: Click the "Record [Valve Name]" button corresponding to the valve you want to record.
Real-time Analysis: During recording, you'll see:
- PCG Plot: A real-time Phonocardiogram of the heart sounds.
- Heart Rate: Current and average heart rate displayed.
- PVC Counter: Number of detected PVCs. (still not working)
- Rhythm and Regularity: Classification of the heart rhythm.
- Animated Heart Icon: A heart icon that beats in sync with the detected heart rate.
- Recording Countdown: A timer indicating the remaining recording time.
- Generate Report: After recording, a PDF report is automatically generated with all relevant data.

5. **SMART Auscultation**
Start SMART Auscultation: Click the "Start SMART Auscultation" button.
Guided Process: The application will guide you through different auscultation points with specific instructions.
Recording and Analysis: At each point, recordings are made, and real-time analysis is performed.
Completion: After all points are completed, a comprehensive PDF report is generated summarizing the findings.

6. **Files Generated**
Audio Recordings: .wav files for each valve recorded.
PCG Plots: .png images of the Phonocardiograms.
Heart Rate Trend Graphs: .png images showing heart rate over time.
RR Intervals Plot: .png images showing RR intervals and detected PVCs.
PDF Reports: Detailed reports including all the above data.

7. **Code Structure**
Global Variables: Variables used throughout the application for state management.

Functions:
- create_directory: Manages the creation of the output directory.
- update_countdown: Updates the recording countdown timer.
- update_plot: Refreshes the PCG plot during recording.
- detect_pvc: Improved function to detect PVCs using median and MAD.
- classify_heart_rhythm: Classifies heart rhythm based on heart rate and intervals.
- bandpass_filter: Applies a bandpass filter to the audio data.
- calculate_heart_rate: Calculates heart rate and detects PVCs.
- plot_rr_intervals: Plots RR intervals and highlights detected PVCs.
- calculate_hrv: Calculates HRV metrics like SDNN and RMSSD.
- animate_heart: Manages the animation of the heart icon.
- generate_pdf_report: Creates a PDF report for individual recordings.
- generate_smart_pdf_report: Creates a comprehensive PDF report for SMART Auscultation.
- record_audio: Handles the audio recording and real-time analysis.
- start_recording: Initiates the recording process.
- start_smart_auscultation: Begins the SMART Auscultation process.
- stop_recording: Stops any ongoing recording.
- save_patient_data: Saves patient information entered in the GUI.

GUI Components:
Tkinter Frames and Widgets: Organized into frames for patient information, valve recordings, SMART auscultation instructions, and real-time visualization.
Styles: Custom styles for consistency in appearance.

Screenshots:

![image](https://github.com/user-attachments/assets/e59b307e-7338-4f80-86c0-22d2a68b2dd2)

![image](https://github.com/user-attachments/assets/7529316b-7225-4b94-bae8-ae1b5a39a087)



