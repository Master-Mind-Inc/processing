from lightgbm import LGBMClassifier
from for_models import slice_for_models, load_prices, insert_predicitions_to_base
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import random
import pandas as pd
import lightgbm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as cm
import numpy as np
import _pickle as cPickle
from accuracy import calculate_accuracy
from cabal_utils.utils import concat_with_window_size, reshape_to_lstm
import keras
from cabal_utils.callbacks import CorrHistoryCallBack
from cabal_utils.utils import profit, concat_with_window_size, reshape_to_lstm
from cabal_utils.losses import get_corr_and_mse_loss
from keras.layers import Input, LSTM, BatchNormalization, RepeatVector,Concatenate, Dropout, Dense, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.initializers import VarianceScaling
from sklearn.metrics import confusion_matrix as cm

from cabal_utils.callbacks import CorrHistoryCallBack
from cabal_utils.utils import profit
from cabal_utils.losses import get_corr_and_mse_loss
from keras.layers import Input, LSTM, RepeatVector,Concatenate, Dropout, Dense, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.initializers import VarianceScaling
from cabal_utils.layers import Attention
import keras
from keras.regularizers import L1L2
from keras.callbacks import LearningRateScheduler as lrs
from keras import backend as K

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection._split import KFold
from itertools import product

keras.layers.Attention = Attention


# os.environ['PATH_CONFIG'] = 'config.json'
# os.environ['PATH_PIPELINE_ID'] = '7'

config = json.load(open(os.environ['PATH_CONFIG']))
params = config["NAMES"]["lgbm"]["PARAMS"]
price = load_prices(config)
pictures_path = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' + 'lgbm_history_fold_'
nn_history_path = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' + 'categorical_nn_history_fold_'

comission = config['COMMISSION'] * 100


def do_get_X(data):
    X = concat_with_window_size(2, data, freq=config['GRANULE'][0])
    X = X.dropna(axis=0)
    return X


def load_X_y(data, target, f):
    X = do_get_X(data)
    y = target.loc[X.index]
    X = reshape_to_lstm(X, 2, f)
    return X, y


def estimate_target_of_point(point, timeframe, percent, dynamic):
/*
------------- UNDER NDA -----------------------
*/
    return breaking_up, breaking_down


def estimate_precise_target_of_point(x):
/*
------------- UNDER NDA -----------------------
*/
    return 0


def short_reward(comm, percent, cm):
    down, flat, up = cm[:, 0]
    return (down * (percent - comm)) - (up * (percent + comm)) - (flat * comm)


def long_reward(comm, percent, cm):
    down, flat, up = cm[:, 2]
    return (up * (percent - comm)) - (down * (percent + comm)) - (flat * comm)


def res_reward(short, long):
    return str(str(int(short)) + ', ' + str(int(long)))


def proba_preds_to_diff(proba, y_test):
    y_pred = (proba * [-100, 0, 100]).sum(axis=1)
    tc = pd.DataFrame(y_pred)
    tc.index = y_test.index
    tc.columns = ['y_pred']
    return tc


def next_price(price, timedelta):
    price['next_ts'] = price['ts'].apply(lambda x: x + pd.Timedelta(timedelta))
    next_price = price['close_price']
    next_price = pd.DataFrame(next_price)
    next_price.columns = ['next_price']
    next_price['next_ts'] = next_price.index
    return pd.merge(price, next_price, on=['next_ts'])


def get_df_plot_acc(name, preds, price, number):
    timedelta = name.split('_')[0]
    n_p = next_price(price, timedelta)
    n_p.index = n_p['ts']
    check = preds.join(pd.DataFrame(n_p['next_price'] - n_p['close_price'])[0], how='inner')
    check.columns = ['y_pred', 'y_test']
    check_ = check.copy()
    thresholds = np.arange(0, 35.1, 0.1)
    p_long, p_short, r_long, r_short = calculate_accuracy('lgbm_acc_' + name, check, thresholds, number)
    return p_long[10], p_long[200], p_short[10], p_short[200], r_long[10], r_long[200], r_short[10], r_short[200]


def create_model():
/*
------------- UNDER NDA -----------------------
*/
    return clf


def create_knn_model():
/*
------------- UNDER NDA -----------------------
*/
    neigh = KNeighborsClassifier(n_neighbors=80)
    return neigh


def create_knn_pca_model():
/*
------------- UNDER NDA -----------------------
*/
    return [pca, neigh]


def create_knn_model_2():
/*
------------- UNDER NDA -----------------------
*/
    return neigh


def create_knn_pca_model_2():
/*
------------- UNDER NDA -----------------------
*/
    return [pca, neigh]


def create_knn_model_3():
/*
------------- UNDER NDA -----------------------
*/
    return neigh


def create_knn_pca_model_3():
/*
------------- UNDER NDA -----------------------
*/
    return [pca, neigh]


# def w_categorical_crossentropy(y_true, y_pred, weights):
#     nb_cl = len(weights)
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=1)
#     y_pred_max = K.expand_dims(y_pred_max, 1)
#     y_pred_max_mat = K.equal(y_pred, y_pred_max)
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#         final_mask += \
#         (K.cast(
#             weights[c_t, c_p]
#             , K.floatx()) *
#                        K.cast(
#                            y_pred_max_mat[:, c_p]
#                            , K.floatx()) *
#                        K.cast(
#                            y_true[:, c_t]
#                            , K.floatx()))
#     return K.categorical_crossentropy(y_pred, y_true) * final_mask


def uncertainty_penalizer_loss(y_true, y_pred, weights):
/*
------------- UNDER NDA -----------------------
*/
    return uncertainty_penalizer_loss(y_true, y_pred, weights=w_array)


def create_rnn_model(f):
/*
------------- UNDER NDA -----------------------
*/
    return model_rnn


def create_american_model(f):
/*
------------- UNDER NDA -----------------------
*/
    return model_am


def lr_schedule(number_of_epoch, learning_rate):
/*
------------- UNDER NDA -----------------------
*/
    return learning_rate


def plot_history(history, name, number):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(nn_history_path + str(number) + '_' + str(name) + '_val_acc.png')
    # summarize history for loss
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(nn_history_path + str(number) + '_' + str(name) + '_val_loss.png')


def analyze_nn_preds(model, x_test_nn):
    preds = model.predict(x_test_nn)
    preds_classes = np.argmax(preds, axis=1)
    preds_classes = preds_classes - 1
    return preds, preds_classes


def fit_model_licv(neigh_2, pca_2, neighs_pca_2,neigh_3, pca_3,
/*
------------- UNDER NDA -----------------------
*/
    return neigh_2, preds_knn_2, rrw_knn_2, pca_2, neighs_pca_2, preds_knn_pca_2, rrw_knn_pca_2, \
           neigh_3, preds_knn_3, rrw_knn_3, pca_3, neighs_pca_3, preds_knn_pca_3, rrw_knn_pca_3


def fit_model(clf, neigh, pca, neigh_pca, model_rnn, model_am, x_train, y_train, x_test, y_test, x_train_nn,
/*
------------- UNDER NDA -----------------------
*/
    return clf, preds, rrw, neigh, preds_knn, rrw_knn, pca, neigh_pca, preds_knn_pca, rrw_knn_pca, \
           model_am, preds_am, rrw_am, model_rnn, preds_rnn, rrw_rnn


def lgbm_preprocess(data_, timeframe, percent, weights=None):
    index_min = data_.index.min()
    index_max = data_.index.max()
    price_ = price[index_min:index_max]

    if config.get('USE_DYNAMIC_TARGET', None):
        dynamic = True
    else:
        dynamic = False

    target = price_.apply(lambda x: estimate_target_of_point(x, timeframe, percent, dynamic), axis=1)
    precise_target = target.apply(lambda x: estimate_precise_target_of_point(x))

    precise_target = pd.DataFrame(precise_target)
    precise_target.columns = ['target']
    features_with_class = data_.join(precise_target, rsuffix='_class', how='inner')
    class_target = features_with_class['target_class']

    features_ = features_with_class.drop(['target', 'target_class'], axis=1)
    if weights is not None:
        features_with_weights = features_.join(weights)
        weights = features_with_weights['tVal']
        features_ = features_with_weights.drop(['tVal'], axis=1)
    print('Features shape: ', features_.shape)
    nn_features, nn_target = load_X_y(features_, class_target, features_.shape[1])
    if weights is not None:
        weights_nn = weights.loc[nn_target.index]

    if weights is None:
        return features_, class_target, nn_features, nn_target
    else:
        return features_, class_target, nn_features, nn_target, weights, weights_nn


def mdi_mda_intersection(X, y, h, p, number):
/*
------------- UNDER NDA -----------------------
*/

    def groupMeanStd(df0, clstrs):
        out = pd.DataFrame(columns=['mean', 'std'])
        for i, j in clstrs.items():
            df1 = df0[j].sum(axis=1)
            out.loc['C_' + str(i), 'mean'] = df1.mean()
            out.loc['C_' + str(i), 'std'] = df1.std() * df1.shape[0] ** -.5
        return out

    def featImpMDI_Clustered(fit, featNames, clstrs):
        df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
        df0 = pd.DataFrame.from_dict(df0, orient='index')
        df0.columns = featNames
        df0 = df0.replace(0, np.nan)
        imp = groupMeanStd(df0, clstrs)
        imp /= imp['mean'].sum()
        return imp

    def featImpMDA_Clustered(clf, X, y, clstrs, n_splits=10):
/*
------------- UNDER NDA -----------------------
*/
        return imp

    def clusterKMeansBase(corr0, maxNumClusters=50, n_init=10):
        x, silh = corr0, pd.Series()
        for init in range(n_init):
            for i in range(30, maxNumClusters + 1):
                kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1)
                kmeans_ = kmeans_.fit(x)
                silh_ = silhouette_samples(x, kmeans_.labels_)
                stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())
                if np.isnan(stat[1]) or stat[0] > stat[1]:
                    silh, kmeans = silh_, kmeans_
        newIdx = np.argsort(kmeans.labels_)
        corr1 = corr0.iloc[newIdx]
        corr1 = corr1.iloc[:, newIdx]
        clstrs = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
        silh = pd.Series(silh, index=x.index)
        return corr1, clstrs, silh

    corr0, clstrs, slh = clusterKMeansBase(X.corr())

    clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                 class_weight='balanced', min_weight_fraction_leaf=0)
    clf = BaggingClassifier(base_estimator=clf, n_estimators=200,
                            max_features=1., max_samples=1., oob_score=False)
    fit = clf.fit(X, y)
    imp = featImpMDI_Clustered(fit, X.columns, clstrs)
    imp_plot = imp.sort_values(['mean'], ascending=False).head(30)[::-1]
    imp_plot.plot(kind='barh', figsize=(20, 10), fontsize=16)

    from random import shuffle
    ke = [int(x.split('_')[1]) for x in
          imp.sort_values(['mean'], ascending=False).head(int(imp.shape[0] / 2))[::-1].index.tolist()]
    print("Ke: ", ke)
    mdi_f = []
    for k in ke:
        l = int(len(clstrs[k]) / 2)
        a = clstrs[k]
        shuffle(a)
        mdi_f.append(a[0:2])
    import itertools
    mdi_f = list(itertools.chain(*mdi_f))

    plt.title(F"MDI Clusters, barrier: ({h}h, {p}%), n_estimators=200", fontsize=22)

    plt.savefig(pictures_path + str(h) + str(p) + str(number) + '_mdi.png')

    clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                 class_weight='balanced', min_weight_fraction_leaf=0)
    clf = BaggingClassifier(base_estimator=clf, n_estimators=10,
                            max_features=1., max_samples=1., oob_score=False)
    imp = featImpMDA_Clustered(clf, X, y, clstrs, 10)
    imp_plot = imp.sort_values(['mean'], ascending=False).tail(30)
    imp_plot.plot(kind='barh', figsize=(20, 10), fontsize=16)
    plt.title(F"MDA Top 20 features, barrier: ({h}h, {p}%), n_estimators=200", fontsize=22)
    plt.savefig(pictures_path + str(h) + str(p) + str(number) + '_mda.png')

    from random import shuffle
    ke = [int(x.split('_')[1]) for x in
          imp.sort_values(['mean'], ascending=False).tail(int(imp.shape[0] / 2)).index.tolist()]
    print("Ke: ", ke)
    mda_f = []
    for k in ke:
        l = int(len(clstrs[k]) / 2)
        a = clstrs[k]
        shuffle(a)
        mda_f.append(a[0:2])
    import itertools
    mda_f = list(itertools.chain(*mda_f))
    print(mdi_f, mda_f, type(mdi_f), type(mda_f))
    print(set(mdi_f), set(mda_f), set(mdi_f) | set(mda_f))
    print(list(set(mdi_f) | set(mda_f)))
    inter = list(set(mdi_f) | set(mda_f))
    return inter, clstrs


def results(preds, number, timeframe, percent, model, name):
    preds['_30m'] = preds.index
    preds = preds[['_30m', "y_pred"]]

    if name == 'lgbm':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'lgbm_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'lgbm_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_lgbm = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'lgbm_{frame}_{percent}.pkl' \
            .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_lgbm, 'wb') as fid:
            cPickle.dump(model, fid)

    if name == 'knn':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'knn|class_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'knn|class_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_knn = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'knn|class_{frame}_{percent}.pkl' \
            .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_knn, 'wb') as fid:
            cPickle.dump(model, fid)
    if name == 'knn_2':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'knn|class_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'licvknn1|class_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_knn = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'licvknn1|class_{frame}_{percent}.pkl' \
            .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_knn, 'wb') as fid:
            cPickle.dump(model, fid)
    if name == 'knn_3':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'knn|class_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'licvknn2|class_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_knn = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'licvknn2|class_{frame}_{percent}.pkl' \
            .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_knn, 'wb') as fid:
            cPickle.dump(model, fid)

    if name == 'knn_pca':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'knnpca|class_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            print("PCA_CLASS")
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'knnpca|class_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_knn_pca = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'knnpca|class_{frame}_{percent}.pkl' \
            .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_knn_pca, 'wb') as fid:
            cPickle.dump(model[1], fid)

        path_pca = os.environ['PATH_PIPELINE_ID'] + '/' + config[
            'PATH_MODELS'] + '/' + 'pca|class_{frame}_{percent}.pkl' \
                           .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_pca, 'wb') as fid:
            cPickle.dump(model[0], fid)

    if name == 'knn_pca_2':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'knnpca|class_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            print("PCA_CLASS")
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'licvknnpca1|class_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_knn_pca = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'licvknnpca1|class_{frame}_{percent}.pkl' \
            .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_knn_pca, 'wb') as fid:
            cPickle.dump(model[1], fid)

        path_pca = os.environ['PATH_PIPELINE_ID'] + '/' + config[
            'PATH_MODELS'] + '/' + 'likvpca1|class_{frame}_{percent}.pkl' \
                           .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_pca, 'wb') as fid:
            cPickle.dump(model[0], fid)

    if name == 'knn_pca_3':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'knnpca|class_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            print("PCA_CLASS")
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'licvknnpca2|class_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_knn_pca = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'licvknnpca2|class_{frame}_{percent}.pkl' \
            .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_knn_pca, 'wb') as fid:
            cPickle.dump(model[1], fid)

        path_pca = os.environ['PATH_PIPELINE_ID'] + '/' + config[
            'PATH_MODELS'] + '/' + 'likvpca2|class_{frame}_{percent}.pkl' \
                           .format(frame=timeframe, percent=str(percent)[2:])
        with open(path_pca, 'wb') as fid:
            cPickle.dump(model[0], fid)

    if name == 'american':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'american|class_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'american|class_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_american = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'american|class_{frame}_{percent}.h5' \
            .format(frame=timeframe, percent=str(percent)[2:])
        model.save(path_american)

    if name == 'rnn':

        if number != len(data) - 1:

            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/'
                + 'rnn|class_{frame}_{percent}_fold_{number}.csv'.
                format(frame=timeframe, percent=str(percent)[2:], number=str(number)),
                index=None)
        else:
            preds.to_csv(
                os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' +
                'rnn|class_{frame}_{percent}.csv'.format(frame=timeframe, percent=str(percent)[2:]),
                index=None)
        path_rnn = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'rnn|class_{frame}_{percent}.h5' \
            .format(frame=timeframe, percent=str(percent)[2:])
        model.save(path_rnn)


data = slice_for_models(config)

rrws = {}

index_min = data[0][0].index.min()
index_max = data[0][0].index.max()
price_ = price[index_min:index_max]
weights_regr = None
if config.get('WEIGHTS', None):
    import statsmodels.api as sm1
    from tqdm import tqdm

    def tValLinR(close):
        x = np.ones((close.shape[0], 2))
        x[:, 1] = np.arange(close.shape[0])
        ols = sm1.OLS(close, x).fit()
        return ols.tvalues[1]


    def getBinsFromTrend(molecule, close, span):
        out = pd.DataFrame(index=molecule, columns=['t1', 'tVal', 'bin'])
        hrzns = range(*span)
        print('WEIGHTS')
        for dt0 in tqdm(molecule):
            df0 = pd.Series()
            iloc0 = close.index.get_loc(dt0)
            if iloc0 + max(hrzns) > close.shape[0]:
                continue
            for hrzn in hrzns:
                dt1 = close.index[iloc0 + hrzn - 1]
                df1 = close.loc[dt0:dt1]
                df0.loc[dt1] = tValLinR(df1.values)
            dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
            out.loc[dt0, ['t1', 'tVal', 'bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1])
            #     out['t1']=pd.to_datetime(out['t1'], unit='s')
        out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
        return out.dropna(subset=['bin'])


    df0 = price_['close_price']
    df1 = getBinsFromTrend(df0.index, df0, [3, 10, 1])
    weights_regr = pd.DataFrame(df0).join(df1['tVal']).fillna(method='ffill').fillna(0)
    weights_regr = np.log(weights_regr['tVal'].abs() + 1)

    datasets_dir = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_DATASETS'] + '/'

    weights_regr.to_csv(datasets_dir + "weights.csv")

for param in params:
    timeframe, percent = param

    for number in range(len(data)):
        train, test = data[number]

        # print(train.shape)
        # print(test.shape)

        print('Start training')
        if config.get('WEIGHTS', None):
            X_train, y_train, X_train_nn, y_train_nn, weights, weights_nn = lgbm_preprocess(train, timeframe, percent, weights_regr)
        else:
            X_train, y_train, X_train_nn, y_train_nn = lgbm_preprocess(train, timeframe, percent)
            weights = None
            weights_nn = None

        X_test, y_test, X_test_nn, y_test_nn = lgbm_preprocess(test, timeframe, percent)
        if config.get('FS', None):
            feats, clstrs = mdi_mda_intersection(X_train, y_train, str(timeframe), str(percent), str(number))
            X_train = X_train[feats]
            X_test = X_test[feats]
            X_train_nn, y_train_nn = load_X_y(X_train, y_train, X_train.shape[1])
            X_test = X_test[feats]
            X_test = X_test[feats]
            X_test_nn, y_test_nn = load_X_y(X_test, y_test, X_test.shape[1])
            config["NAMES"]['lgbm'][str(timeframe) + '_' + str(percent)[2:]] = feats
            clstrs_config = {}
            for k in clstrs.keys():
                clstrs_config[str(k)] = clstrs[k]

            config['clstrs'] = clstrs_config
        f = X_train.shape[1]
        my_pca, neighs_pca = create_knn_pca_model()
        model, preds, rrw, neigh, preds_knn, rrw_knn, my_pca_fit, neighs_pca_fit, preds_knn_pca, rrw_knn_pca, \
         model_am, preds_am, rrw_am, model_rnn, preds_rnn, rrw_rnn = \
            fit_model(create_model(),
                      create_knn_model(),
                      my_pca,
                      neighs_pca,
                      create_american_model(f),
                      create_rnn_model(f),
                      X_train, y_train,
                      X_test, y_test,
                      X_train_nn, y_train_nn,
                      X_test_nn, y_test_nn,
                      number,
                      timeframe,
                      percent,
                      weights, weights_nn)

    # print(pl_1, pl_20, ps_1, ps_20, rl_1, rl_20, rs_1, rs_20 )

        results(preds_knn, number, timeframe, percent, neigh, name='knn')
        results(preds_knn, number, timeframe, percent, [my_pca_fit, neighs_pca_fit], name='knn_pca')

        results(preds, number, timeframe, percent, model, name='lgbm')
        results(preds, number, timeframe, percent, model_am, name='american')
        results(preds, number, timeframe, percent, model_rnn, name='rnn')

        rrws[str(timeframe) + '_' + str(percent) + '_' + str(number)] = rrw
        rrws[str(timeframe) + '_' + str(percent) + '_american_' + str(number)] = rrw_am
        rrws[str(timeframe) + '_' + str(percent) + '_rnn_' + str(number)] = rrw_rnn
        rrws[str(timeframe) + '_' + str(percent) + '_knn_' + str(number)] = rrw_knn
        rrws[str(timeframe) + '_' + str(percent) + '_knn_pca_' + str(number)] = rrw_knn_pca
        K.clear_session()

print(rrws)

with open(os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_PREDICTIONS'] + '/' + 'rrw.json', 'w') as fp:
    json.dump(rrws, fp)

if config.get("FS", None):
    json.dump(config, open(os.environ['PATH_CONFIG'], 'w'), sort_keys=True, indent=4, separators=(',', ': '))

print('LGBM executed')
