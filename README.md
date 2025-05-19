
# Traning Hand Gesture Recognition Using TensorFlow Lite

This repository contains a project for hand gesture recognition using data from an accelerometer and gyroscope. The data is pre-processed, normalized, and used to train a model with TensorFlow. The trained model is then converted to TensorFlow Lite for deployment on embedded systems.

## 📑 Table of Contents

- [Project Structure](#project-structure)  
- [Key Features](#key-features)  
- [Use Cases](#use-cases)  
- [Getting Started](#getting-started)  
  - [Normalizing the Data](#normalizing-the-data)  
  - [How Normalization Works in This Project](#how-normalization-works-in-this-project)  
- [Training the Model](#training-the-model)  
  - [How Training Works](#how-training-works)  
  - [Data Loading](#data-loading)  
  - [Model Architecture](#model-architecture)  
  - [Training](#training)  
  - [Model Evaluation](#model-evaluation)  
  - [Saving the Model](#saving-the-model)  
- [Converting the Model to TensorFlow Lite](#converting-the-model-to-tensorflow-lite)  
  - [Conversion Process](#conversion-process)  
  - [Optimization](#optimization)  
  - [Saving the TFLite Model](#saving-the-tflite-model)

---
## Project Structure

- `data/`: Contains raw sensor data and the processed/normalized data.
- `models/`: Contains the trained TensorFlow Lite model.
- `scripts/`: Contains ipynb scripts for normalizing data and training the model.
- `grapth/`: contain the acuraccy and loss graphs for the tf model

### Key Features:
- Data pre-processing and normalization for accelerometer and gyroscope sensor data.
- Hand gesture classification using TensorFlow.
- Conversion of the trained model to TensorFlow Lite format for embedded systems.
- Python scripts for data normalization, training, and model conversion.
  
### Use Cases:
- Real-time hand gesture recognition on embedded devices.
- Gesture-based control systems for IoT and wearable applications.

## Getting Started

### Normalizing the Data

The raw data collected from the accelerometer and gyroscope sensors typically has different ranges and scales. To ensure that the model learns efficiently, it's important to normalize the data. Normalization helps to standardize the data and bring all features into the same scale, which can improve the performance and convergence of machine learning algorithms.
### How Normalization Works in This Project

1. **Data Input**: The raw data is collected and stored in CSV files within the `data/raw_data/` folder.
   - The data includes features from the accelerometer (x, y, z axes) and gyroscope (x, y, z axes).
   - The labels represent the different hand gestures.

2. **Normalization Process**:
   - The `normalize_data.py` script uses **StandardScaler** from the `sklearn.preprocessing` module to normalize the features (accelerometer and gyroscope data). 
   - The `StandardScaler` performs **Z-score normalization**, where the data is transformed to have a mean of 0 and a standard deviation of 1. This ensures that each feature (e.g., `accel_x`, `gyro_x`) contributes equally to the model's training.
   - The labels (gesture types) remain unchanged.

3. **Output**: After normalization, the processed data is saved in the `data/processed_data/` folder in a CSV format. The normalized data is ready for use in model training.


## Training the Model

Once the data is normalized, we can proceed to train a model for hand gesture recognition. In this project, we use **TensorFlow** to build and train a neural network.

### How Training Works
open the colab file to follow along
1. **Data Loading**:
- The training dataset is split into training and validation sets using **train_test_split** from `sklearn`.
X = df[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']]  # Features
y = df['Gesture']  # Target (ensure it's encoded beforehand)
   - The features ('Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz') are separated from the labels (gesture (circle and circl AWC).
   - The labels are encoded using **LabelEncoder** to convert the gesture labels (e.g., 'circle', 'circle AWC') into binary value (0 mean circle an 1 is circl AWC).

2. **Model Architecture**:
   - A simple neural network model is built using **Keras** (part of TensorFlow). The model consists of:
     - An input layer that matches the shape of the feature vector (6 features: accelerometer and gyroscope axes).
     - A couple of fully connected hidden layers with **ReLU** activation for non-linearity.
     - An output layer with **softmax** activation to predict the probability distribution of different gesture classes.

3. **Training**:
   - The model is trained using **sparse categorical cross-entropy loss** (since the labels are integer encoded) and the **Adam optimizer**.
   - The training dataset is split into training and validation sets using **train_test_split** from `sklearn`.

4. **Model Evaluation**:
   - After training, the model is evaluated on the validation set to check its performance.
   - The accuracy score and loss are printed to the console, which helps to assess the model's performance.

5. **Saving the Model**:
   - The trained model is saved in both **Keras format** (`.h5`) and **TensorFlow Lite format** (`.tflite`).
   - The TensorFlow Lite model is specifically designed for deployment on embedded devices, making it lightweight and optimized for edge computing applications.

## Converting the Model to TensorFlow Lite

After training the model, it can be converted to **TensorFlow Lite (TFLite)** format to be deployed on embedded systems, such as microcontrollers or mobile devices. TensorFlow Lite is an optimized version of TensorFlow that enables models to run efficiently on resource-constrained devices.

### How to Convert the Model to TFLite

1. **Conversion Process**:
   - Once the model is trained and saved in Keras format (`.h5`), the next step is to convert it to TensorFlow Lite format using TensorFlow’s `TFLiteConverter`.
   - The `convert_to_tflite.py` script is responsible for performing this conversion.

2. **Optimization**:
   - During conversion, the model can be optimized to reduce its size and improve inference speed without sacrificing much accuracy. The script supports various optimizations such as quantization, which reduces the precision of the model's weights to save memory.
   - By default, the model is saved with full precision (float32). However, optional optimization techniques, like **post-training quantization**, can be applied to further optimize the model.

3. **Saving the TFLite Model**:
   - The converted model is saved as a `.tflite` file, which is much smaller and optimized for mobile and embedded device use.



