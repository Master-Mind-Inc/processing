import json
import os
import gc
import pandas as pd
import numpy as np 
np.random.seed(42)
from for_models import slice_for_models
from for_models import insert_predicitions_to_base

import tensorflow as tf
tf.set_random_seed(42)

from accuracy import calculate_accuracy

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from cabal_utils.utils import *
# from cabal_utils.fitgenerator import *

import keras
from keras import backend as K
from keras.models import load_model
from keras.models import Model, Sequential

from keras.layers import LSTM, Dense, CuDNNLSTM, CuDNNGRU, Dropout,Conv1D, MaxPooling1D, TimeDistributed

from keras import losses
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.regularizers import L1L2


# os.environ['PATH_CONFIG'] = 'config.json'
# os.environ['PATH_PIPELINE_ID'] = '4'
config = json.load(open(os.environ['PATH_CONFIG']))

model_name = 'rnn'

def plot_history(history, corr_callback):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    corr = corr_callback.correlations

    # Visualize loss history
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(111)
    ax1.plot(training_loss, '--')
    ax1.plot(test_loss, '-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Test Loss'], loc='upper left')
    ax1.set_yscale('log')

    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(corr, 'g-')
    ax2.set_ylabel('Corr')
    ax2.legend(['Corr'], loc='upper right')

    # plt.savefig('ETH_model_corr_0.07966_features_1234567RSI_BTC_features_126_window_size_5.png')
    plt.grid()
    plt.show()




comission = 0.003
window_size = 2
test_size= 0.2
shuffle_data = False





def do_get_X(data):
    data = data.drop(['target'], axis=1)
    data = data[features]
    data = data.copy()
    X = concat_with_window_size(window_size, data, freq=config['GRANULE'][0])
    X = X.dropna(axis = 0)
    return X

def do_get_target(data):
    return data['target']

def load_X_y(data):
    X = do_get_X(data)
    y = do_get_target(data).loc[X.index]
    X = reshape_to_lstm(X,  window_size, f)
    return X, y




def load_dataset(name):
    data = pd.read_csv(name, index_col = 0)
    return index_to_datetime(data, periods = 5)





from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
/*
------------- UNDER NDA -----------------------
*/

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config['step_dim'] = self.step_dim
        return config


    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim



from cabal_utils.callbacks import CorrHistoryCallBack
from cabal_utils.utils import profit
from cabal_utils.losses import get_corr_and_mse_loss
from keras.layers import Input, LSTM, RepeatVector,Concatenate, Dropout, Dense, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.initializers import VarianceScaling

def create_model(): 
/*
------------- UNDER NDA -----------------------
*/
    return model



def get_sample_waight(y):
    l = len(y_train.index)
    betha = 100000
    sample_weight = (np.exp(np.arange(l)/betha)/betha)
    sample_weight = sample_weight/sample_weight.max()*4
    return sample_weight


def save_plot_history(history, corr_callback, number):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    corr = corr_callback.correlations

    # Visualize loss history
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(111)
    ax1.plot(training_loss, '--')
    ax1.plot(test_loss, '-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Test Loss'], loc='upper left')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(corr, 'g-')
    ax2.set_ylabel('Corr')
    ax2.legend(['Corr'], loc='upper right')

    # plt.savefig('ETH_model_corr_0.07966_features_1234567RSI_BTC_features_126_window_size_5.png')
    plt.grid()
    plt.savefig(os.environ['PATH_PIPELINE_ID']+'/'+config['PATH_PREDICTIONS']+'/'+'rnn_fold_'+str(number)+'_history.png')

    
def fit_model(model,
/*
------------- UNDER NDA -----------------------
*/
    return  model if not save_best_model else load_model('saved_best_model.h5')





def save_plot_test_pred(y_test, y_pred, number):
    y_test = np.reshape(y_test, (-1))
    y_pred = np.reshape(y_pred, (-1))
    print('Samples :', len(y_pred))
    print(np.corrcoef(y_test, y_pred))
    plt.figure(figsize = (10,10))
    plt.scatter(y_test, y_pred, s=2)
    size = np.max(np.abs([y_test, y_pred]))
    plt.plot([-size, size], [0, 0], c='black', linewidth=1.0)
    plt.plot([0, 0], [-size, size], c='black', linewidth=1.0)
    plt.plot([-0.5 * size, 0.5 * size], [-0.5 * size, 0.5 * size], '--', c='goldenrod', linewidth=1.0)
    plt.axis('equal')
    plt.xlabel('Real course change(%)')
    plt.ylabel('Predicted course change(%)')
    plt.savefig(os.environ['PATH_PIPELINE_ID']+'/'+config['PATH_PREDICTIONS']+'/' +
                'rnn_fold_' + str(number) + '.png')

def results(model, number):
    print('Saving predictions')
    y_pred = model.predict(X_test).reshape(-1)
    print('Correlation: ', np.corrcoef(y_pred, y_test))
    save_plot_test_pred(y_test, y_pred, number)

    check_acc = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
    name = 'rnn_acc'
    thresholds = np.arange(0, 35.1, 0.1)
    _, _, _, _ = calculate_accuracy(name, check_acc, thresholds, number)
    if number != len(data) - 1:
        res_pred = pd.DataFrame({'_30m': y_test.index, 'y_pred': y_pred})
        pd.DataFrame({'_30m':y_test.index, 'y_pred': y_pred}).to_csv(
            os.environ['PATH_PIPELINE_ID']+'/'+config['PATH_PREDICTIONS']+'/'+'rnn_fold_' + str(number) +'.csv',
                index=False)
    else:
        res_pred = pd.DataFrame({'_30m': y_test.index, 'y_pred': y_pred})
        pd.DataFrame({'_30m':y_test.index, 'y_pred': y_pred}).to_csv(
        os.environ['PATH_PIPELINE_ID']+'/'+config['PATH_PREDICTIONS']+'/'+'rnn.csv',
                    index=False)

    
    model.save(os.environ['PATH_PIPELINE_ID']+'/'+config['PATH_MODELS']+'/'+'rnn'+'.h5')
    return np.corrcoef(y_pred, y_test)[0][1], res_pred
    
def analyze_correlation(corrs, config):
    dataset_dir = os.environ['PATH_PIPELINE_ID']+'/'+config['PATH_PREDICTIONS']+'/'
    if len(corrs) == 1:
        print('One fold correlation')
        pd.DataFrame({'corr': [corrs[0]]}).to_csv(dataset_dir + 'corr_rnn.csv')
    else:
        corrs = np.array(corrs)
        filter_corr = corrs.mean() - 2 * corrs.std()
        print(str(len(corrs)) + ' Folds')
        print('Mean - 2 * std: ' + str(filter_corr))
        print(corrs)
        pd.DataFrame({'corr': corrs}).to_csv(dataset_dir + 'corr_rnn.csv')
        pd.DataFrame({'filter': [filter_corr]}).to_csv(dataset_dir + 'filter_corr_rnn.csv')


data = slice_for_models(config)

corrs = []
preds = []
for number in range(len(data)):
    train, test = data[number]
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
    print('RNN features', features)
    f = len(features)

    print(train.shape)
    print(test.shape)

    print('Start training')
    model = create_model()

    X_train, y_train = load_X_y(train)
    X_test, y_test = load_X_y(test)

    model = fit_model(model, X_train, y_train,
                      X_test, y_test, number,
                      epochs=config['NAMES']['rnn']['EPOCHS'])

    corr, pred = results(model, number)
    corrs.append(corr)
    preds.append(pred)

    K.clear_session()


analyze_correlation(corrs, config)

if "PREDICTIONS_TO_BASE" in config:
    if config['PREDICTIONS_TO_BASE']:
        insert_predicitions_to_base(config, preds, "3")

print('RNN executed')
