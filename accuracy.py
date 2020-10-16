import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cm
import os, json

# os.environ['PATH_CONFIG'] = 'config.json'
# os.environ['PATH_PIPELINE_ID'] = '7'

config = json.load(open(os.environ['PATH_CONFIG']))


pictures_path = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'


def calculate_accuracy(name, test_df, thresholds, number, draw_accuracy_plot=True):
/*
------------- UNDER NDA -----------------------
*/

    return list_of_accuracies_long, list_of_accuracies_short, list_of_accuracies_recall_long, \
           list_of_accuracies_recall_short