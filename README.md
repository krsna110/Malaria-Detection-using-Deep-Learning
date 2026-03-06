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



