'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import numpy as np
from collections import Counter


def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Your code here
    model_pred_df = pd.read_csv("data/prediction_model_03.csv")
    genres_df = pd.read_csv("data/genres.csv") 
    return model_pred_df,genres_df


model_pred_df,genres_df = load_data()

def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''
    #Getting genre_list (list)
    #Create list of unique genres
    genre_list = genres_df['genre'].tolist()

    #Getting genre_true_counts (dict)
    #Convert actual genres column to string
    myString = model_pred_df["actual genres"].to_string(index=False)

    #Clean column into list
    myString = myString.replace("[","")
    myString = myString.replace("]","")
    myString= myString.replace("'","")
    myString= myString.replace(" ","")
    myString= myString.replace("\n",",")
    myList = list(myString.split(","))

    #Load list into counter
    myCounter = Counter(myList)
    #print(myCounter)

    #Convert counter to dict
    genre_true_counts = dict(myCounter)
    #print(genre_true_counts)

    #Getting genre_tp_counts (dict)
    #Get correct rows
    correctRows = model_pred_df[model_pred_df['correct?'] == 1]
    #print(correctRows)
  
    #Convert actual genres column to string
    myString = correctRows["actual genres"].to_string(index=False)

    #Clean column into list
    myString = myString.replace("[","")
    myString = myString.replace("]","")
    myString= myString.replace("'","")
    myString= myString.replace(" ","")
    myString= myString.replace("\n",",")
    myList = list(myString.split(","))

    #Load list into counter
    myCounter = Counter(myList)
    #print(myCounter)

    #Convert counter to dict
    genre_tp_counts = dict(myCounter)
    #print(genre_tp_counts)

    # Getting genre_fp_counts (dict) 
    #Get incorrect rows
    incorrectRows = model_pred_df[model_pred_df['correct?'] == 0]
    
  
    #Convert actual genres column to string
    myString = incorrectRows["actual genres"].to_string(index=False)

    #Clean column into list
    myString = myString.replace("[","")
    myString = myString.replace("]","")
    myString= myString.replace("'","")
    myString= myString.replace(" ","")
    myString= myString.replace("\n",",")
    myList = list(myString.split(","))

    #Load list into counter
    myCounter = Counter(myList)
    #print(myCounter)

    #Convert counter to dict
    genre_fp_counts = dict(myCounter)
    #print(genre_fp_counts)

    return genre_list,genre_true_counts,genre_tp_counts,genre_fp_counts
    # Your code here
process_data(model_pred_df,genres_df)