Dataset available - https://drive.google.com/file/d/1JwOx6JpM1GY60__gj25jrQiSzKZZCeF9/view?usp=sharing


#Problem Statement:
You are required to develop a web based or mobile application that can interphase with the microphone on the device and provide visual feedback to the user based on the word that was spoken.


The model at a later stage is expected to run on an edge device with limited resources, you are expected to design the model in such a way that it can be deployed on a Raspberry Pi 3 or a STM32H747II. 


Data set:
We have been provided with exemplars of the wake words "door open," "door stop," and "door close."
Our Approach:


We are using a CNN model to recognize wake words related to door actions (specifically, "door open," "door stop," and "door close") using audio data. The overall approach involves the following steps:


Feature Extraction:
The extract_mfcc_features function is used to extract Mel-frequency cepstral coefficients (MFCC) features from raw audio data. MFCC features are commonly employed in speech and audio processing tasks.


Data Preprocessing:
The preprocess_data function loads audio data from a specified directory, applies a bandpass filter (butter_bandpass_filter), extracts MFCC features for each audio file, and splits the data into training and testing sets.


Model Architecture:
A Convolutional Neural Network (CNN) model is defined in the train_model function using the Keras framework. The model is designed to take input in the form of MFCC features and includes convolutional layers, max-pooling layers, dropout layers, and dense layers.


Model Training:
The defined CNN model is trained using the extracted MFCC features. Training is performed on the training set (X_train, y_train) with a specified number of epochs and batch size.






Model Evaluation:
The trained model is evaluated on the test set (X_test, y_test) using various performance metrics such as accuracy, precision, recall, and F1 score. The confusion matrix is generated and visualized to provide insights into the model's classification performance.


Results Display:
The code concludes with the display of the shapes of MFCC features and the model input, as well as the evaluation results including accuracy, precision, recall, F1 score, and the confusion matrix.


Technical Challenges                                                
Extremely diverse data conditions: The model is expected to perform with a high accuracy on a diverse set of environmental conditions such as areas with background noise and the audio patterns of a diverse gender and pronunciations of the same word.                
Compute Limitation: Being that the model must be deployed on a low compute device the model is expected to utilize under 8MB of RAM and 16MB of ROM at any given time. You are expected to produce some kind of proof to back up the claims that the model is only utilizing the above mentioned resources.        
Algorithm Efficiency: The time taken to perform a prediction must be less than the length of the audio used for the inference to be considered realtime.                                         
                                
#Libraries :
os: The os module provides a way to interact with the operating system, and it's likely used here to handle file paths and directory operations.
librosa: Librosa is a Python package for audio and music analysis. It's commonly used for loading audio files, extracting features, and performing various audio processing tasks.
numpy as np: NumPy is a fundamental package for scientific computing with Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these elements. The np alias is a common convention.
from sklearn.model_selection import train_test_split: The train_test_split function from scikit-learn is used to split a dataset into training and testing sets. This is crucial for evaluating the performance of a machine learning model.
from sklearn.preprocessing import LabelEncoder: Label encoding is a technique used to convert categorical labels into numerical format, which is required for many machine learning algorithms. The LabelEncoder from scikit-learn helps with this conversion.
from sklearn.ensemble import RandomForestClassifier: Random Forest is an ensemble learning method for classification, regression, and other tasks. In this code, it seems a Random Forest classifier will be used for the audio classification task.
from sklearn.metrics import accuracy_score: This imports the accuracy_score metric from scikit-learn, which will be used to evaluate the accuracy of the machine learning model.
from python_speech_features import mfcc: This import statement brings in the Mel-frequency cepstral coefficients (MFCC) function from the python_speech_features library. MFCCs are commonly used features in audio signal processing, especially in speech and audio recognition tasks.
from tqdm import tqdm: TQDM is a library for adding progress bars to loops and other iterable computations. It's used to visualize the progress of tasks, and in this code, it may be applied to show progress during some processing steps.
Scipy.signal -  The scipy.signal module is part of the SciPy library, which is an open-source library used for scientific and technical computing in Python. The scipy.signal module specifically focuses on signal processing operations. It provides various functions and tools for working with signals, systems, and related operations.


IPython.display - The IPython.display module provides tools for displaying various types of content within an IPython environment, such as Jupyter notebooks or IPython interactive sessions. The Audio class in IPython.display is specifically designed for displaying audio content, allowing you to embed audio playback directly into your Jupyter notebooks or interactive environments.


 Tqdm library - This is a fast, extensible progress bar for Python and is often used to visualize the progress of tasks, especially when dealing with large datasets or time-consuming computations.
* 







#Extracting MFCC features:


The purpose of this section is to define a function that extracts Mel-frequency cepstral coefficients (MFCC) features from an audio signal, facilitating feature extraction for tasks such as speech and audio processing.


Parameters we are taking in extract_mfcc_features function


audio :
* Type: 1D array
* Purpose: Represents the input audio signal from which MFCC features will be extracted.


 sr=16000:
* Type: Integer, optional (default: 16000)
* Purpose: Sampling rate of the audio signal. It determines the number of samples of audio per second.
num_cepstral=13:
* Type: Integer, optional (default: 13)
* Purpose: Number of cepstral coefficients to be extracted. MFCCs are represented by a set of coefficients that capture the spectral characteristics of the audio signal.
frame_length=0.02:
* Type: Float, optional (default: 0.02)
* Purpose: Length of the analysis window in seconds. This parameter determines the duration of each frame used in the feature extraction process.
frame_stride=0.02:
   * Type: Float, optional (default: 0.02)
   * Purpose: Stride or step size between frames in seconds. It specifies the time shift between consecutive frames.
num_filters=32 :
   * Type: Integer, optional (default: 32)
   * Purpose: Number of mel filters used in the mel spectrogram computation. Mel filters are applied to the power spectrum to create a compressed representation.
fft_length=320:
* Type: Integer, optional (default: 320)
* Purpose: Length of the Fast Fourier Transform (FFT) window. It determines the number of points used in the FFT computation, influencing the frequency resolution of the STFT.
preemphasis_coeff=0.98):
* Type: Float, optional (default: 0.98)
* Purpose: Coefficient for pre-emphasis filtering. Pre-emphasis boosts higher frequencies to enhance the signal quality by compensating for the roll-off in spectral energy.
Steps involve in MFCC extraction: 


Apply pre-emphasis - Pre-emphasis is a high-pass filtering technique that boosts the higher frequencies in the signal.


Short-Time Fourier Transform (STFT):  computes the Short-Time Fourier Transform (STFT) of the pre-emphasized audio signal using the specified parameters such as hop length, FFT length, and window type (“hand”)


Mel Spectrogram:  calculates the mel spectrogram from the squared magnitude of the STFT result. The mel spectrogram represents the energy distribution across different frequency bands.
CODE - “mel_spectrogram = librosa.feature.melspectrogram(S=np.abs(stft_result)**2, sr=sr, n_mels=num_filters)”
After this we will apply log on the mel_spectrogram


MFCC Extraction:  extracts the Mel-frequency cepstral coefficients (MFCC) from the log mel energy using Discrete Cosine Transform (DCT). MFCCs are widely used in audio processing and speech recognition. Once its done do Mean-Variance Normalization. And return mdcc_features.

#Preprocessing of the data:


The purpose of the code is to preprocess audio data from a specified directory, extracting MFCC features, applying a bandpass filter, and organizing the data for machine learning tasks by splitting it into training and testing sets.
import the tqdm library for displaying progress bars and librosa for audio processing.
Then we will  defines a function named preprocess_data that takes a directory path (data_dir) containing audio files organized in subdirectories.


Inside our function :
1. Frist - store raw audio data (data), corresponding labels (labels), and extracted features (features).


2. Now iterates over each subdirectory (class) in the specified data_dir, then iterates over each audio file in that class. It loads each audio file using librosa.load with a specified sampling rate of 16,000 Hz. We will iterates over each subdirectory (class) in the specified data_dir to organize the data based on different classes or categories.


3. The raw audio data along with the corresponding label is appended to the data list.


4. The raw audio data is passed through a bandpass filter (butter_bandpass_filter) and then the MFCC features are extracted (extract_mfcc_features) from the filtered audio.


5. The extracted features and labels are appended to the respective lists.


6. The lists of features and labels are converted to NumPy arrays for further processing for compatibility with numerical operations.


7. The data is split into training and testing sets using train_test_split from scikit-learn.


8. The function returns the processed features, classes, and split training and testing sets.


9. directory path is provided, and the function can be called with this path to preprocess the audio data.




#CNN training:


compiles, and trains a Convolutional Neural Network (CNN) model for sound classification using the provided MFCC (Mel-frequency cepstral coefficients) features and corresponding labels, and then saves the trained model.
* Label Encoding:
   * The code starts by using scikit-learn's LabelEncoder to convert categorical class labels (y_train and y_test) into numerical labels. This is a common preprocessing step in machine learning.
   * * Model Architecture:
   * - The model is a sequential neural network with the following layers:
   * - Input: Takes the input shape of the MFCC features.
   * - Reshape: Reshapes the input to have a single channel (useful for convolutional layers).
   * - Conv2D: Applies convolutional layers with 8 filters in the first layer and 16 filters in the second layer, both using a 3x3 kernel and the ReLU activation function.
   * - MaxPooling2D: Performs max pooling to reduce spatial dimensions.
   * - Dropout: Applies dropout regularization to prevent overfitting.
   * - Flatten: Flattens the output to a 1D array for the fully connected layers.
   * - Dense: A fully connected layer with a softmax activation function for classification.
* * Compilation:
   * - The model is compiled using the Adam optimizer, sparse categorical crossentropy as the loss function (suitable for integer-encoded class labels), and accuracy as the evaluation metric.
* * Training:
   * - The model is trained using the fit method on the training data (X_train and encoded_labels_train) for 100 epochs with a batch size of 32. Validation data (X_test and encoded_labels_test) is used to monitor the model's performance during training.
* * Model Saving:
   * - After training, the model is saved to a file named "custom_model.keras."
* * Returning Model and Encoder:
   * - The trained model and the label encoder are returned from the function.
* * Data Preprocessing:
   * - Features, classes, and training/testing data are assumed to be obtained from a preprocess_data function.
* * * * Print Shapes:
   * - The shapes of the MFCC features and the model input are printed to provide insights into the data dimensions.
* * Training the Model:
   * - The train_model function is called with the provided data, and the trained model and label encoder are stored in the variables trained_model and label_encoder.








Evaluate Model:


Here we defines a function, evaluate_model, which takes a trained machine learning model, test data, and label encoder, predicts labels on the test set, calculates and prints accuracy, precision, recall, and F1 score, and generates a visual representation of the confusion matrix using seaborn and matplotlib.
- Defining a function named evaluate_model that takes a trained model (model), test features (X_test), true labels (y_test), and a label encoder (label_encoder) as input.


- Using the trained model to make predictions on the test set (X_test).


- Converting the predicted class probabilities to class labels using the inverse transformation of the label encoder. It is necessary because the output of a neural network for multi-class classification is often in the form of predicted probabilities for each class. The predicted probabilities represent the likelihood or confidence of the model that a given sample belongs to each class.


- Calculating and printing the accuracy of the model on the test set.


- Calculating and printing precision, recall, and F1 score using weighted averaging.


- Generating and plotting the confusion matrix using seaborn. The confusion matrix provides a visual representation of how well the model is classifying each class.


Copyright © 2024 neural Navigators. All rights reserved.
