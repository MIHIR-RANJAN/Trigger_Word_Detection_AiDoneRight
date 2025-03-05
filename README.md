# Trigger Word Detection - AiDoneRight

## Overview
Trigger Word Detection is a web-based application designed for hands-free interaction using voice commands. The application detects three wake words—**"door open," "door stop," and "door close"**—and provides visual feedback. The model is optimized for deployment on **low-power edge devices** such as **Raspberry Pi 3** and **STM32H747II**, ensuring real-time performance under limited computational resources.

## Features
- **Real-Time Voice Command Recognition**: Detects specific trigger words with high accuracy.
- **Web-Based Interface**: Built using **ReactJS**, allowing seamless microphone access and server communication.
- **Optimized for Edge Devices**: Designed to run on low-memory devices (≤ 8MB RAM, ≤ 16MB ROM).
- **Robust Audio Processing**: Utilizes **MFCC feature extraction** and **bandpass filtering** to improve speech recognition in noisy environments.
- **Fast and Efficient Predictions**: Inference time is shorter than the audio length, ensuring real-time responsiveness.

## Dataset
### Dataset available - https://drive.google.com/file/d/1JwOx6JpM1GY60__gj25jrQiSzKZZCeF9/view?usp=sharing

## Technical Stack
### **Frontend**
- **ReactJS** for UI development.
- **Web Audio API** for microphone access and recording.
- **WebSocket (socket.io-client)** for real-time communication with the backend.
- **Dynamic UI Feedback** based on recognized trigger words.

### **Backend**
- **Flask + Flask_SocketIO** for real-time server communication.
- **Librosa** for audio loading and feature extraction.
- **Pre-trained CNN Model** for classification based on **MFCC features**.
- **WebSocket** for streaming audio data to the server.

### **Model Architecture**
#### **CNN Model** (Primary Model)
- **Two Conv2D layers** with ReLU activation and MaxPooling.
- **Dropout layers (0.25)** for regularization.
- **Fully connected (Dense) layers** with Softmax activation.
- **Adam optimizer** and **Sparse Categorical Crossentropy** loss function.
- **Performance Metrics**:
  - **Accuracy**: 95.56%
  - **Precision**: 95.62%
  - **Recall**: 95.56%
  - **F1 Score**: 95.57%

#### **LSTM Model** (Alternative Model)
- **Single LSTM Layer** with 64 units, ReLU activation, and L2 regularization.
- **Dropout layer (0.2)** for regularization.
- **Dense layer** with **Sigmoid activation**.
- **Performance Metrics**:
  - **Accuracy**: 93.37%
  - **Precision**: 93.48%
  - **Recall**: 93.37%
  - **F1 Score**: 93.35%

## **Audio Processing Pipeline**
### **MFCC Feature Extraction**
1. **Pre-emphasis Filtering**: Boosts higher frequencies to enhance signal quality.
2. **Short-Time Fourier Transform (STFT)**: Computes time-frequency representation.
3. **Mel Spectrogram**: Extracts spectral energy across different frequency bands.
4. **Log Transformation**: Enhances feature stability.
5. **Discrete Cosine Transform (DCT)**: Extracts MFCC coefficients.
6. **Mean-Variance Normalization**: Standardizes extracted features.

## **Installation & Setup**
### **Prerequisites**
- Python 3.8+
- Node.js & npm
- Flask & Flask-SocketIO
- Required Python Libraries:
  ```bash
  pip install numpy librosa scikit-learn python_speech_features scipy tqdm tensorflow flask flask-socketio
  ```

### **Running the Application**
1. **Start the Backend Server**:
   ```bash
   python server.py
   ```
2. **Start the Frontend Application**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

## **Edge Device Deployment**
- Optimized for **Raspberry Pi 3**.
- Uses lightweight CNN architecture for efficient processing.
- Memory footprint ≤ 8MB RAM, ≤ 16MB ROM.
- Can run inference in real-time.

## **Use Cases**
- **Home Automation**: Voice-activated control for smart home devices.
- **In-Car Voice Control**: Safer hands-free interactions while driving.
- **IoT Integration**: Compatible with embedded systems for various automation tasks.

## **Live Demo**
A live demo is available, demonstrating real-time trigger word detection and visual feedback.

## **Contributors**
- **Aashish**
- **Shashi**
- **Mihir**
- **Kathyayani**
- **Snigdha**

## **License**
© 2024 Neural Navigators. All rights reserved.

## **Additional Information**
### **Executive Summary**
- **Application Focus**: Web app for touch-free interaction with single-button activation.
- **Functionality**: Identifies three door-related commands and provides visual feedback.
- **Edge Device Deployment**: Model optimized for low-power devices, with focus on Raspberry Pi.
- **User-Friendly Design**: Simple button-driven interface for accessibility.

### **Solution Overview**
- **Webpage**: Uses **WebSocket** for real-time server communication.
- **Backend**: Saves audio as `.wav` files, extracts **MFCC features**, and classifies input using a **pre-trained model**.
- **Audio Preprocessing**: Uses **STFT** for frequency analysis and **Mel spectrograms** for feature extraction.

### **Architecture & Live Demo**
- **Real-time integration** with Raspberry Pi for practical applications.
- **Confusion matrix analysis** provides insights into model performance.
- **In-depth evaluation** ensures high accuracy and robustness across diverse environments.

### **Conclusion**
The versatility of this project makes it applicable for multiple domains:
- **Home Automation**: Enables users to control smart devices seamlessly.
- **Automotive Applications**: Enhances in-car voice control systems.
- **IoT & Embedded Systems**: Can be integrated into a variety of real-world applications.
