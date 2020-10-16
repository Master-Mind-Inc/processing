import os
import json
import pandas as pd
import pickle 

import gc
import numpy as np 
np.random.seed(42)


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from cabal_utils.utils import *
from cabal_utils.utils import plot_history


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import time 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

import _pickle as cPickle
from for_models import slice_for_models
from for_models import insert_predicitions_to_base

# os.environ['PATH_PIPELINE_ID'] = '7'
# os.environ['PATH_CONFIG'] = 'config.json'

from accuracy import calculate_accuracy

config = json.load(open(os.environ['PATH_CONFIG']))

print('Starting knn_pca')


def save_predictions(X_test, y_test, y_pred, number):

    predictions = pd.DataFrame(data = {'_30m':y_test.index, 
                                       'y_pred':y_pred})
    check_acc = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
    thresholds = np.arange(0, 35.1, 0.1)
    _, _, _, _ = calculate_accuracy('knn_pca_acc', check_acc, thresholds, number)
    if number != len(data) - 1:
        predictions.to_csv(
            os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' + 'knn_pca_fold_'+str(number)+'.csv',
            index=False)
    else:
        predictions.to_csv(
            os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' + 'knn_pca.csv',
            index=False)
        


data = slice_for_models(config)
corrs = []
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
    print('knn_pca features', features)



    pca_corrs = {}
    for pca_comp in config['NAMES']['knn_pca']['PCA_COMP']:

        X_train, y_train = train[features].values, train['target']

        X_test, y_test = test[features].values, test['target']

        np.random.seed(42)

        pca = PCA(n_components=pca_comp)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        neigh = KNeighborsRegressor(n_neighbors=config['NAMES']['knn_pca']['NEIGH'])
        neigh.fit(X_train, y_train)
        pred = []
        for x in (X_test):
            pred.append(neigh.predict(x.reshape(1, -1)))# .reshape(1, pca_comp)))

        pred = np.array(pred).reshape(y_test.shape[0], )
        print('pca_comp:',pca_comp)
        corr = np.corrcoef(y_test.values, pred)[0][1]
        print('corr:',corr)
        pca_corrs[pca_comp] = corr
    corrs.append(pca_corrs)

    
corrs = pd.DataFrame(corrs)

corrs.to_csv(os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' + 'pca_corrs.csv')
if corrs.shape[0] > 1:
    final_comp = (corrs.mean(axis=0) - 2 * corrs.std(axis=0)).idxmax()
    max = (corrs.mean(axis=0) - 2 * corrs.std(axis=0)).max()
    pd.DataFrame({final_comp: [max]})\
        .to_csv(os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' + 'filter_pca_corrs.csv')
else:
    final_comp = corrs.T.idxmax()[0]
    max = corrs.T.max()[0]
    pd.DataFrame({final_comp: [max]}) \
        .to_csv(os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' + 'filter_pca_corrs.csv')

preds = []
for iterator in range(len(data)):
    train = data[iterator][0]
    test = data[iterator][1]

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

    X_train, y_train = train[features].values, train['target']

    X_test, y_test = test[features].values, test['target']

    np.random.seed(42)

    pca = PCA(n_components=final_comp)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    neigh = KNeighborsRegressor(n_neighbors=config['NAMES']['knn_pca']['NEIGH'])
    neigh.fit(X_train, y_train)
    pred = []
    for x in (X_test):
        pred.append(neigh.predict(x.reshape(1, -1)))

    pred = np.array(pred).reshape(y_test.shape[0], )
    print('final_comp', final_comp)
    corr = np.corrcoef(y_test.values, pred)[0][1]
    print('corr:', corr)

    save_predictions(X_test, y_test, pred.reshape(pred.shape[0], ), iterator)

    pred_for_base = pd.DataFrame({'_30m': y_test.index, 'y_pred': pred.reshape(pred.shape[0], )})
    preds.append(pred_for_base)

filename = 'pca.pkl'
path = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + filename

with open(path, 'wb') as fid:
    cPickle.dump(pca, fid)

filename_knn = 'knn_pca.pkl'
path_knn = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + filename_knn

with open(path_knn, 'wb') as fid:
    cPickle.dump(neigh, fid)

if "PREDICTIONS_TO_BASE" in config:
    if config['PREDICTIONS_TO_BASE']:
        insert_predicitions_to_base(config, preds, "2")

print('KNN_PCA executed')


