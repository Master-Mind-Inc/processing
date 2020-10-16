import os
import json
import pandas as pd

import gc
import numpy as np 
np.random.seed(42)
from for_models import insert_predicitions_to_base
from accuracy import calculate_accuracy


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from cabal_utils.utils import *
from cabal_utils.utils import plot_history
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import time 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import _pickle as cPickle
from for_models import slice_for_models


# os.environ['PATH_CONFIG'] = 'config.json'
# os.environ['PATH_PIPELINE_ID'] = '7'
config = json.load(open(os.environ['PATH_CONFIG']))


print('Starting KNN')


def save_predictions(X_test, y_test, y_pred, name, number):

    predictions = pd.DataFrame(data = {'_30m':y_test.index, 
                                       'y_pred':y_pred})

    predictions.to_csv(os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                       name, index=False)

    check_acc = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
    thresholds = np.arange(0, 35.1, 0.1)
    _, _, _, _ = calculate_accuracy('knn_acc', check_acc, thresholds, number)


data = slice_for_models(config)

corrs = []
preds = []

for number in range(len(data)):
    train = data[number][0]
    test = data[number][1]
    features = list(train.filter(like='feature').columns) + \
                  list(train.filter(like='bbe').columns) + \
                   list(train.filter(like='dema').columns) + \
                   list(train.filter(like='normed').columns)
    if 'USE_BCH_FEATURES' in config:
        if config['USE_BCH_FEATURES']:
            features += list(train.filter(like='bf').columns)
    if 'USE_STAKAN_FEATURES' in config:
        if config['USE_STAKAN_FEATURES']:
            features += list(train.filter(like='stakan').columns)
    if 'USE_SWING_FEATURES' in config:
        if config['USE_SWING_FEATURES']:
            features += list(train.filter(like='swing').columns)
    if 'DIVERGENCE' in config:
        features += list(train.filter(like='divergence').columns)
    if 'PUMP' in config:
        features += list(train.filter(like='pump').columns)

    print('knn features', features)

    X_train, y_train = train[features].values, train['target']

    X_test, y_test = test[features].values, test['target']

    np.random.seed(42)

    neigh = KNeighborsRegressor(n_neighbors=config['NAMES']['knn']['NEIGH'])
    neigh.fit(X_train, y_train)
    pred = []
    for x in (X_test):
        pred.append(neigh.predict(x.reshape(1, -1)))# .reshape(1, pca_comp)))

    pred = np.array(pred).reshape(y_test.shape[0], )
    
    if number != len(data) - 1:
        preds_csv_name = 'knn_fold_' + str(number) + '.csv'
    else:
        preds_csv_name = 'knn.csv'

    save_predictions(X_test, y_test, pred.reshape(pred.shape[0], ), preds_csv_name, number)

    pred_for_base = pd.DataFrame({'_30m': y_test.index, 'y_pred': pred.reshape(pred.shape[0], )})
    preds.append(pred_for_base)

    print(np.corrcoef(y_test.values, pred))

    corrs.append(np.corrcoef(y_test.values, pred)[0][1])

    filename = 'knn.pkl'
    path = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + filename
    with open(path, 'wb') as fid:
        cPickle.dump(neigh, fid)


def analyze_correlation(corrs, config):
    dataset_dir = os.environ['PATH_PIPELINE_ID']+'/'+config['PATH_PREDICTIONS']+'/'
    if len(corrs) == 1:
        print('One fold correlation')
        pd.DataFrame({'corr': [corrs[0]]}).to_csv(dataset_dir + 'corr_knn.csv')
    else:
        corrs = np.array(corrs)
        filter_corr = corrs.mean() - 2 * corrs.std()
        print(str(len(corrs)) + ' Folds')
        print('Mean - 2 * std: ' + str(filter_corr))
        print(corrs)
        pd.DataFrame({'corr': corrs}).to_csv(dataset_dir + 'corr_knn.csv')
        pd.DataFrame({'filter': [filter_corr]}).to_csv(dataset_dir + 'filter_corr_knn.csv')


analyze_correlation(corrs, config)

if "PREDICTIONS_TO_BASE" in config:
    if config['PREDICTIONS_TO_BASE']:
        insert_predicitions_to_base(config, preds, "1")

print('KNN executed')