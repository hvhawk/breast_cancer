# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 03:27:42 2021

@author: harsh
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
mdl=pickle.load(open(r"D:\College Projects\Jupyter Projects\small projekt 1\q1\svm_trained.sav",'rb'))

# creating a function for prediction

def breast_cancer(input_data):
    #changing the input data to numpy array
    inp_dat_np=np.asarray(input_data)
    
    #reshaping the array as we are predicting for one instance
    inp_data=inp_dat_np.reshape(1,-1)
    
    predict=mdl.predict(inp_data)
    print(predict)
    
    if(predict[0]==0):
        return "Benign" 
    else:
        return "Malignant"
    
def main():
    # giving a title
    st.title("Breast Cancer Prediction")
    
    #getting input data from the user
    #radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst,
    
    radius_mean = st.text_input("Mean Radius")
    texture_mean = st.text_input("Mean Texture")
    perimeter_mean = st.text_input("Mean Perimeter")
    area_mean = st.text_input("Mean Area")
    smoothness_mean = st.text_input("Mean Smoothness")
    compactness_mean = st.text_input("Mean Compactness")
    concavity_mean = st.text_input("Mean Concavity")
    concave_points_mean = st.text_input("Mean Concave Points")
    symmetry_mean = st.text_input("Mean Symmetry")
    fractal_dimension_mean = st.text_input("Mean Fractal Dimension")
    radius_se = st.text_input("SE Radius")
    texture_se = st.text_input("SE Texture")
    perimeter_se = st.text_input("SE Perimeter")
    area_se = st.text_input("SE Area")
    smoothness_se = st.text_input("SE Smoothness")
    compactness_se = st.text_input("SE Compactness")
    concavity_se = st.text_input("SE Concavity")
    concave_points_se = st.text_input("SE Concave Points")
    symmetry_se = st.text_input("SE Symmetry")
    fractal_dimension_se = st.text_input("SE Fractal Dimension")
    radius_worst = st.text_input("Worst Radius")
    texture_worst = st.text_input("Worst Texture")
    perimeter_worst = st.text_input("Worst Perimeter")
    area_worst = st.text_input("Worst Area")
    smoothness_worst = st.text_input("Worst Smoothness")
    compactness_worst = st.text_input("Worst Compactness")
    concavity_worst = st.text_input("Worst Concavity")
    concave_points_worst = st.text_input("Worst Concave Points")
    symmetry_worst = st.text_input("Worst Symmetry")
    fractal_dimension_worst = st.text_input("Worst Fractal Dimension")
    
    # code for Prediction
    diagnosis = ''
    
    #creating button for prediction
    if st.button("Breast Cancer Type Result"):
        diagnosis = breast_cancer([radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst])
    
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    