+++
date = "2025-03-28"
draft = false
title = "Getting Started with Kaggle - Hands-on Handwritten Digit Recognition"
image = "title.jpg"
categories = ["Deep Learning"]
tags = ["Kaggle","Models","AI","CNN"]

copyright="疏间徒泍"

+++

# Deep Learning Basics - Kaggle Digit Recognizer Tutorial

**Table of Contents**

[TOC]

## Introduction

> **[Kaggle](https://www.aigc.cn/kaggle)** is a premier platform in data science, founded in 2010 by Anthony Goldbloom in Melbourne and later acquired by Google Cloud in 2017. It serves as a bridge connecting organizations needing data solutions with skilled data professionals through crowdsourcing competitions. With over 800,000 active data scientists, Kaggle allows companies to post challenges with datasets and monetary rewards, enabling participants to develop innovative solutions.

This tutorial will guide you through building a **Convolutional Neural Network (CNN)** for the classic **Digit Recognizer** competition on Kaggle. Designed for absolute beginners, we'll cover the entire workflow—from data processing to model deployment—while explaining core deep learning concepts.

**Prerequisite:** Ensure you have a Kaggle account. For registration guidance, visit: [Kaggle Account Setup Guide](https://blog.csdn.net/weixin_51288849/article/details/130164188).

## Step 1: Understand the Competition

Visit the competition page: [[Digit Recognizer]](https://www.kaggle.com/competitions/digit-recognizer/). Key sections to review:
- **Overview**: Competition objectives and rules.
- **Data**: Dataset structure details.

> **Dataset Notes:**  
> - Each 28x28 grayscale image is flattened into a 784-pixel vector (0-255 values).  
> - `train.csv` contains labels in the first column and pixel values in subsequent columns.  
> - Test images (`test.csv`) lack labels and require prediction.

## Step 2: Data Preprocessing

**Create a Kaggle Notebook** via the competition page's *Submit Prediction* > *Notebook*. Begin with essential imports:

```python
# ========== Part 1: Import Libraries ==========
import numpy as np          
import pandas as pd         
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
import tensorflow as tf     
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Input, MaxPooling2D  
```

**Load and Process Data:**  
```python
# Load datasets
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv') 
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# Split features and labels
X_train = train_data.drop('label', axis=1).values  
y_train = train_data['label'].values               
X_test = test_data.values                          

# Normalize pixel values (0-1 range)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to 28x28x1 images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Split training data into training/validation sets (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```

## Step 3: Build the CNN Model

We adapt the classic **LeNet-5** architecture:

```python
model = Sequential([
    Input((28, 28, 1)),  # Input layer
    Conv2D(6, (5,5), activation='sigmoid', padding='valid'),  # Feature extraction
    AveragePooling2D((2,2)),  # Downsampling
    Conv2D(16, (5,5), activation='relu', padding='valid'),  
    MaxPool2D((2,2)),  
    Flatten(),  # Transition to dense layers
    Dense(120, activation='relu'),  
    Dropout(0.3),  # Regularization
    Dense(84, activation='relu'),  
    Dense(10, activation='softmax')  # Output layer
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

View model architecture with `model.summary()`.

## Step 4: Train the Model

```python
history = model.fit(
    X_train, y_train,
    epochs=10,        
    batch_size=32,    
    validation_data=(X_val, y_val)
)
```

**Sample Output:**  
```
Epoch 10/10
1050/1050 ━━━ 2s 2ms/step - accuracy: 0.9775 - loss: 0.0718 - val_accuracy: 0.9758 - val_loss: 0.0809
```

## Step 5: Evaluate Performance

Visualize training metrics:

```python
# Plot accuracy and loss curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.show()
```

## Step 6: Generate Predictions

```python
# Predict test set labels
predictions = model.predict(X_test) 
predicted_labels = np.argmax(predictions, axis=1)

# Create submission file
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels)+1),
    'Label': predicted_labels
})
submission.to_csv('submission.csv', index=False)
```

## Step 7: Submit to Kaggle

1. Download `submission.csv` from the notebook.  
2. Navigate to the competition's *Submit Predictions* page.  
3. Upload the file and check your public score (~98% accuracy).  

**Congratulations! 🎉** You've completed your first Kaggle competition!

## Step 8: Performance Optimization (Optional)

- **Increase Epochs**: Train longer (e.g., 30 epochs).  
- **Data Augmentation**: Use `ImageDataGenerator` for synthetic data.  
- **Model Tuning**: Replace activation functions, add layers, or adopt advanced architectures like ResNet.  
- **Hyperparameter Tuning**: Adjust learning rates, dropout rates, etc.  

## Step 9: Local Deployment

Export the model and build a GUI app for real-time predictions:

```python
# Save model
model.save('digit_recognizer_model.h5')

# Install dependencies locally
# Run: pip install tensorflow pillow numpy
```

Use the provided `digit_recognizer_app.py` script to create an interactive drawing interface.

---

## Conclusion

The Digit Recognizer project is the "Hello World" of deep learning, offering a practical introduction to CNNs and Kaggle workflows. Keep experimenting with advanced techniques to sharpen your skills! 🚀

## Appendix: Complete Code

```python
# ========== COMPLETE PIPELINE ==========
# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, AveragePooling2D, MaxPool2D, 
    Flatten, Dense, Dropout, Input
)

# ----- Data Loading -----
print("\n[1/6] Loading and preprocessing data...")
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# ----- Data Preparation -----
# Separate features and labels
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.values

# Normalize pixel values (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to 28x28x1 images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Split into training/validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# ----- Model Construction -----
print("\n[2/6] Building LeNet-5 architecture...")
model = Sequential([
    Input(shape=(28, 28, 1)),
    
    # First convolution block
    Conv2D(6, (5,5), activation='sigmoid', padding='valid'),
    AveragePooling2D((2,2)),
    
    # Second convolution block
    Conv2D(16, (5,5), activation='relu', padding='valid'),
    MaxPool2D((2,2)),
    
    # Classification head
    Flatten(),
    Dense(120, activation='relu'),
    Dropout(0.3),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ----- Model Training -----
print("\n[3/6] Training model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ----- Performance Visualization -----
print("\n[4/6] Generating training curves...")
plt.figure(figsize=(12,4))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# ----- Model Evaluation -----
print("\n[5/6] Evaluating model...")
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# ----- Prediction & Submission -----
print("\n[6/6] Generating predictions...")
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Create submission file
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels)+1),
    'Label': predicted_labels
})
submission.to_csv('submission.csv', index=False)

print("\n✔ Pipeline completed successfully!")
print("Submission file saved as 'submission.csv'")
```