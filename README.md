# Malaria-Detection-using-Deep-Learning

\# Malaria Detection using Deep Learning



\## Activity 2.1: Importing the Libraries



In this step, the required Python libraries were imported to perform data processing, visualization, and machine learning operations.



\### Libraries Used



\*\*NumPy\*\*

NumPy is used for numerical computations and mathematical operations on arrays.



\*\*Pandas\*\*

Pandas is used for data manipulation and analysis using data structures such as DataFrames.



\*\*Matplotlib\*\*

Matplotlib is used to create visualizations such as graphs and plots to understand the dataset.



\*\*Seaborn\*\*

Seaborn is an advanced visualization library built on top of Matplotlib that helps create attractive statistical plots.



\*\*Train\_Test\_Split\*\*

This function is used to divide the dataset into training data and testing data.



\*\*Confusion Matrix\*\*

The confusion matrix is used to evaluate the performance of a classification model.



\*\*StandardScaler\*\*

StandardScaler is used to normalize the data so that all features have a similar scale.



\*\*Joblib\*\*

Joblib is used to save and load trained machine learning models.



\### Importing Libraries Code



```python

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model\_selection import train\_test\_split

from sklearn.metrics import confusion\_matrix, accuracy\_score

from sklearn.preprocessing import StandardScaler



import tensorflow as tf

import joblib

```



\### Conclusion



All the required libraries were successfully imported to support data analysis, preprocessing, visualization, and deep learning model development for malaria detection.



\## Activity 2.2: Reading the Dataset

In this step, the malaria cell image dataset was loaded and prepared for training the deep learning model.

The dataset consists of microscopic images of red blood cells that are categorized into two classes: **Parasitized** and **Uninfected**.

Reading the dataset is an important step because the model must first access and understand the input data before learning patterns from it. The images are stored in folders and are loaded into the program using TensorFlow’s data preprocessing tools.



\### Dataset Description

\*\*Parasitized\*\*

This folder contains images of red blood cells that are infected with malaria parasites.


\*\*Uninfected\*\*

This folder contains images of healthy red blood cells without malaria infection.

The dataset is organized into two main directories:


\*\*Training Dataset\*\*

Used for training the deep learning model.


\*\*Validation Dataset\*\*

Used to evaluate the performance of the model during training.


\### Reading Dataset Code

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Creating Image Data Generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Loading Training Dataset
train_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

# Loading Validation Dataset
val_generator = val_datagen.flow_from_directory(
    "dataset/val",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"

```

\###Conclusion

The malaria dataset was successfully read using the ImageDataGenerator and flow_from_directory() functions. The images were resized, normalized, and prepared in batches so they can be used efficiently for training the deep learning model.



\## Activity 4: Saving the Model

After training the deep learning model, it is important to save the trained model so that it can be reused later without retraining.

Saving the model allows us to load it directly in the web application and perform predictions on new blood cell images.

In this project, the trained model is saved using the `model.save()` function from TensorFlow Keras. The model is stored in **HDF5 format (.h5)** which contains the model architecture, learned weights, and training configuration.



\### Saving Model Code

```python
import tensorflow as tf

# Save the trained model
model.save("malaria_model.h5")

print("Model saved successfully as malaria_model.h5")
```
