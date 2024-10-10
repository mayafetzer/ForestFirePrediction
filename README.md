# Project: Forest Fire Prediction

## Author
Maya Fetzer  
Semester: Fall 2024  
Course: CHEG 472  

## Purpose
This repository creates an app to model if a forest fire with occur.

## Public App
Here is the public app for the forest fire prediction model: https://huggingface.co/spaces/mayafetzer/ForestFires

## Files in this repository
ForestFirePrediction.ipynb -  Google Colab code used to preprocess the dataset and create the machine learning model. 
nn
wildfire_prediction_multi_output_dataset_v2.xlsx - An example dataset used to train the machine learning model. 

app.py - Python code used to create the Hugging Face app

best_model_fire_duration.pkl - A pickle file of the best model for the fire duration

best_model_fire_occurrence.pkl - A pickle file of the best model for fire occurrence

best_model_fire_size.pkl - A pickle file of the best model for fire size
     
best_model_suppression_cost.pkl - A pickle file for the best model for supression cost   
 
scaler.pkl - A pickle file to scale the input values for the machine learning model.  
       
requirements.txt - The requirements to run the app in Hugging Face

## Prerequisites

### Python
Ensure you have Python 3.10 or later installed.

### Libraries
Install the following libraries using pip:

```
pip install gradio
pip install pandas
pip install numpy
pip install sklearn
pip install openpyxl
pip install pickle5
```

## Explanation

- **Streamlit**: Provides a simple way to create interactive web applications with Python.
- **Matplotlib**: Used for creating visualizations like plots and charts.
- **Pandas**: Offers data structures and analysis tools for working with tabular data.
- **NumPy**: Provides efficient numerical operations and arrays.
- **Sklearn**: Machine learning library.
- **Gradio:** A user-friendly library for building and sharing interactive web interfaces for machine learning models and data science projects.
- **Openpyxl:** A library for reading and writing Excel files in the XLSX format, making it easy to work with spreadsheets directly from Python.
- **Pickle5:** An enhanced version of Python's pickle module for object serialization and
