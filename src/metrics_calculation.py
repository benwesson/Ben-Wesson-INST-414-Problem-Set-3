'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Your code here
    tp_list = []
    fp_list = []

    #Sum true positives
    for item in genre_tp_counts:
        tp_list.append(genre_tp_counts[item])
    tp_sum = sum(tp_list)
    #print(tp_sum)
    
    #Sum false positives
    for item in genre_fp_counts:
        fp_list.append(genre_fp_counts[item])
    fp_sum = sum(fp_list)
    #print(fp_sum)

    #Calcualte precision
    micro_precision = tp_sum / (tp_sum + fp_sum)
    #print(micro_precision)

    #I am guessing that the blank genres are false negatives otherwise its zero
    #Calculate recall 
    micro_recall = tp_sum / (tp_sum + genre_fp_counts[''])
    #print(micro_recall)

    #Calcualte F1 score
    top = (micro_precision * micro_recall)
    bottom =  (micro_precision + micro_recall)
    micro_f1 = (2 * top) / (2 * bottom) 
    #print(micro_f1)

    mytuple = (micro_precision, micro_recall, micro_f1)
    macro_prec_list = 1
    macro_recall_list = 1
    macro_f1_list = 1
    #return mytuple,
    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list 
    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    # Your code here
