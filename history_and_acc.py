import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from sklearn.pipeline import Pipeline
from cabal_utils.layers import Attention
from cabal_utils.losses import get_corr_and_mse_loss
import pickle
from keras.models import load_model
from cabal_utils.utils import *
import json
import re
from tqdm import tqdm

import pandas as pd
from oracle_data.DataLoader import DatabaseLoader
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cm
import os, json

from oracle_data.DataLoader import DatasetBuilder
import keras
import tensorflow as tf
from keras import backend as K
from clickhouse_driver import Client
import os
import sys


def uncertainty_penalizer_loss(y_true, y_pred, weights):
    return K.mean(
        K.sum
        ((K.cast(
                K.gather(K.constant(weights), K.argmax(y_true, axis=1)),
                 dtype='float32') * K.cast(y_pred, dtype='float32')), axis=1))


w_array = np.array([[0, 1, 2], [1.5, 0 ,1.5], [2, 1, 0]])


def w_loss(y_true, y_pred):
    return uncertainty_penalizer_loss(y_true, y_pred, weights=w_array)


keras.layers.Attention = Attention
keras.losses.corr_loss_and_mse = get_corr_and_mse_loss(1, 100)
keras.losses.w_loss = w_loss
import os
from pathlib import Path
import lightgbm
import requests

client = Client(password="password",
                host="host",
                user="default",
                port=int(29000))

# config = json.load(open(os.environ['PATH_CONFIG']))
config = json.load(open('./config.json'))
config['PHASE'] = '1min'

binance_username = 'default'
binance_server = 'db-host'
binance_port = 8123
binance_quote = 'USDT'
binance_password = 'password'

################## Эти параметры задаются извне ############################
pipe_name = config['PATH_PIPELINE_ID']
# pipe_name = '/3000_'

pipe_id = pipe_name.split('/')[-1].split('_')[0]

if config.get('MMI_MODE', None):
    pipe_id = pipe_id + 'm'



############################################################################

def init_names():
    names = []
    for name in config['NAMES']:
        if name == 'lgbm':
            for li in config['NAMES'][name]['PARAMS']:
                timeframe, percent = li
                percent = str(percent)[2:]
                names.append('{name}_{timeframe}_{percent}.pkl'.format(name=name,
                                                                       timeframe=timeframe,
                                                                       percent=percent))
                names.append('{name}_{timeframe}_{percent}.h5'.format(name='rnn|class',
                                                                      timeframe=timeframe,
                                                                      percent=percent))
                names.append('{name}_{timeframe}_{percent}.h5'.format(name='american|class',
                                                                      timeframe=timeframe,
                                                                      percent=percent))
                names.append('{name}_{timeframe}_{percent}.pkl'.format(name='knn|class',
                                                                       timeframe=timeframe,
                                                                       percent=percent))
                names.append('{name}_{timeframe}_{percent}.pkl'.format(name='knnpca|class',
                                                                       timeframe=timeframe,
                                                                       percent=percent))
        elif (name == 'american_v4') | (name == 'rnn'):
            names.append(name)
        else:
            names.append(name)
    return names


def load_models():
    models = {}
    for name in names:
        if name.split('_')[0] == 'lgbm':
            n, t, p = name.split('_')
            t = re.search('[0-9]+', t).group()
            if len(t) == 1:
                t = '0' + t
            db_key = models_tbls[name.split('_')[0]] + t[:2] + p.strip('.pkl')[:3]
            instance = pickle.load(open('./models/{name}'.format(name=name), 'rb'))
            models[name.split('_')[0] + '_' + db_key] = instance

        elif name.split('_')[0] == 'rnn|class':
            n, t, p = name.split('_')
            t = re.search('[0-9]+', t).group()
            if len(t) == 1:
                t = '0' + t
            db_key = models_tbls['rnn'] + t[:2] + p[:-3][:3]
            instance = load_model('./models/{name}'.format(name=name))
            instance._make_predict_function()
            models[name.split('_')[0] + '_' + db_key] = instance

        elif name.split('_')[0] == 'american|class':
            n, t, p = name.split('_')
            t = re.search('[0-9]+', t).group()
            if len(t) == 1:
                t = '0' + t
            db_key = models_tbls['american_v4'] + t[:2] + p[:-3][:3]
            instance = load_model('./models/{name}'.format(name=name))
            instance._make_predict_function()
            models[name.split('_')[0] + '_' + db_key] = instance

        elif name.split('_')[0] == 'knn|class':
            n, t, p = name.split('_')
            t = re.search('[0-9]+', t).group()
            if len(t) == 1:
                t = '0' + t
            db_key = models_tbls['knn'] + t[:2] + p.strip('.pkl')[:3]
            instance = pickle.load(open('./models/{name}'.format(name=name), 'rb'))
            instance.n_jobs = 1
            models[name.split('_')[0] + '_' + db_key] = instance

        elif name.split('_')[0] == 'knnpca|class':
            n, t, p = name.split('_')
            t = re.search('[0-9]+', t).group()
            if len(t) == 1:
                t = '0' + t
            db_key = models_tbls['knn_pca'] + t[:2] + p.strip('.pkl')[:3]
            instance_pca_class = pickle.load(open('./models/{name}'.format(name=name[3:]), 'rb'))
            instance_knn_class = pickle.load(open('./models/{name}'.format(name=name), 'rb'))
            instance_knn_class.n_jobs = 1
            instance = Pipeline([('pca', instance_pca_class), ('knn', instance_knn_class)])
            models[name.split('_')[0] + '_' + db_key] = instance

        elif name == 'american_v4':
            instance = load_model('./models/american_v4.h5')
            instance._make_predict_function()
            models[name + '_' + models_tbls[name]] = instance
        elif name == 'rnn':
            instance = load_model('./models/rnn.h5')
            instance._make_predict_function()
            models[name + '_' + models_tbls[name]] = instance
        elif name == 'knn':
            instance = pickle.load(open('./models/knn.pkl', 'rb'))
            instance.n_jobs = 1
            print(instance.predict)
            models[name + '_' + models_tbls[name]] = instance
        elif name == 'knn_pca':
            instance_pca = pickle.load(open('./models/pca.pkl', 'rb'))
            instance_knn = pickle.load(open('./models/knn_pca.pkl', 'rb'))
            instance_knn.n_jobs = 1
            print(instance_knn.predict)
            instance = Pipeline([('pca', instance_pca), ('knn', instance_knn)])
            models[name + '_' + models_tbls[name]] = instance
    return models


def get_features_for_models():
/*
------------- UNDER NDA -----------------------
*/
    return features_dict


def get_predictions(models_dict, features_dict, date_starts_per_model):
    predictions_dict = {}
    for key in models_dict.keys():
        if key.split('_')[0] == 'lgbm':
            features = features_dict['lgbm']
            try:
                date_st = date_starts_per_model[key.split('_')[1]]
                if date_st == 'collector':
                    continue
                features = features[date_st:]
            except:
                pass
            if config.get('FS', None):
                names = config['NAMES']['lgbm'][str(int(key.split('_')[1][1:3])) + 'h' + '_' + key.split('_')[1][-2:]]
                features = features[names]
            instance = models_dict[key]
            y_pred = (instance.predict_proba(features) * [-100, 0, 100]).sum(axis=1)
            tc = pd.DataFrame(y_pred)
            tc.index = features.index
            tc.columns = ['price_del']
            predictions_dict[key] = tc
        if (key.split('_')[0] == 'knn|class') | (key.split('_')[0] == 'knnpca|class'):
            features = features_dict['lgbm']
            try:
                date_st = date_starts_per_model[key.split('_')[1]]
                if date_st == 'collector':
                    continue
                features = features[date_st:]
            except:
                pass
            if config.get('FS', None):
                names = config['NAMES']['lgbm'][str(int(key.split('_')[1][1:3])) + 'h' + '_' + key.split('_')[1][-2:]]
                features = features[names]
            instance = models_dict[key]
            y_pred = (instance.predict_proba(features) * [-100, 0, 100]).sum(axis=1)

            if config.get('WEIGHTS', None):
                datasets_dir = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_DATASETS'] + '/'
                weights = pd.read_csv(datasets_dir + "weights.csv", parse_dates=True,
                                      index_col=[0], header=None)
                try:
                    indexes = instance.kneighbors(features, return_distance=False)
                except:
                    indexes = instance[1].kneighbors(instance[0].transform(features), return_distance=False)
                mean_w = weights[1][indexes].mean(axis=1)
                y_pred = y_pred * mean_w
            tc = pd.DataFrame(y_pred)
            tc.index = features.index
            tc.columns = ['price_del']
            predictions_dict[key] = tc
        if key.split('_')[0] == 'knn':
            features = features_dict['knn']
            try:
                date_st = date_starts_per_model[key.split('_')[1]]
                if date_st == 'collector':
                    continue
                features = features[date_st:]
            except:
                pass
            instance = models_dict[key]
            instance.n_jobs = 1
            y_pred = instance.predict(features)
            tc = pd.DataFrame(y_pred)
            tc.index = features.index
            tc.columns = ['price_del']
            predictions_dict[key] = tc
            if config['NAMES']['knn'].get('LEARNING_DISTANCE'):
                y_pred = instance.kneighbors(features)[0].mean(axis=1)
                tc = pd.DataFrame(y_pred)
                tc.index = features.index
                tc.columns = ['price_del']
                predictions_dict['ld' + '_' + models_tbls['ld']] = tc
        if (key.split('_')[0] == 'rnn') | (key.split('_')[0] == 'american'):
            index, features = features_dict['nn']
            instance = models_dict[key]
            y_pred = instance.predict(features)
            tc = pd.DataFrame(y_pred)
            tc.index = index
            tc.columns = ['price_del']
            predictions_dict[key] = tc

        if (key.split('_')[0] == 'rnn|class') | (key.split('_')[0] == 'american|class'):
            features = features_dict['lgbm']
            try:
                date_st = date_starts_per_model[key.split('_')[1]]
                if date_st == 'collector':
                    continue
                features = features[date_st:]
            except:
                pass
            if config.get('FS', None):
                names = config['NAMES']['lgbm'][str(int(key.split('_')[1][1:3])) + 'h' + '_' + key.split('_')[1][-2:]]
                features = features[names]
                X = concat_with_window_size(window_size=2, data=features, freq=config['GRANULE'][0]).dropna()
                index = X.index
                features = reshape_to_lstm(X, 2, -1)
            else:
                X = concat_with_window_size(window_size=2, data=features, freq=config['GRANULE'][0]).dropna()
                index = X.index
                features = reshape_to_lstm(X, 2, -1)

            instance = models_dict[key]
            y_pred = (instance.predict(features) * [-100, 0, 100]).sum(axis=1)
            tc = pd.DataFrame(y_pred)
            tc.index = index
            tc.columns = ['price_del']
            predictions_dict[key] = tc
    return predictions_dict


def create_queries_and_insert(exchange, pipe_number, model_number, dataset):
    create_db_query = "CREATE DATABASE IF NOT EXISTS db_{exchange};"
    create_db_query = create_db_query.format(exchange=exchange)

    create_logs_query = '''
    CREATE TABLE IF NOT EXISTS db_{exchange}.tbl_{pipe_name}_{pipe_model}_logs(
    ts UInt64,
    price Float64,
    price_del Float64
    ) ENGINE = MergeTree() ORDER BY (ts) SETTINGS index_granularity = 8192;
    '''

    create_logs_query = create_logs_query.format(exchange=exchange, pipe_name=pipe_number, pipe_model=model_number)

    print(create_db_query, file=sys.stderr)
    print(create_logs_query, file=sys.stderr)

    client.execute(create_db_query)

    client.execute(create_logs_query)

    insert_query = 'INSERT INTO db_{exchange}.tbl_{pipe_name}_{pipe_model}_logs (ts, price, price_del) VALUES'
    insert_query = insert_query.format(exchange=exchange, pipe_name=pipe_number, pipe_model=model_number)
    print(insert_query, file=sys.stderr)

    list_to_insert = list(zip(dataset['ts'].values.tolist(),
                              dataset['price'].values.tolist(),
                              dataset['price_del'].values.tolist()))

    client.execute(insert_query, list_to_insert)

    return 0


def insert_all_in_db(predictions_dict):
    for key in predictions_dict.keys():
        df = predictions_dict[key].join(prices)
        df = df.reset_index()
        df['ts'] = df['index'].apply(lambda x: int(pd.Timestamp(x).timestamp()))
        df = df.drop(['index'], axis=1)
        create_queries_and_insert('binance', pipe_id, key.split('_')[-1], df)
    return 0


w_sizes = ['8h', '24h', '72h', '120h']
max_window_size = w_sizes[-1]


def receive(x, test_df):
    piece = test_df[x['ts'] - pd.Timedelta(max_window_size): x['ts']]
    return piece


def acc_(ttd):
    _acc = acc(ttd['y_test_sign'], ttd['y_pred_sign'])

    if np.isnan(_acc):
        _acc = 0
    return _acc


def calculate_one_ws(test_df, test_df_long, test_df_short, sign, perc):
/*
------------- UNDER NDA -----------------------
*/
    return accuracy_, accuracy_long, accuracy_short, recall_, recall_long, recall_short


def calculate_accuracy(test_df, sign):
/*
------------- UNDER NDA -----------------------
*/

    return /*
------------- UNDER NDA -----------------------
*/



def get_target_and_accuracy(test_df):
/*
------------- UNDER NDA -----------------------
*/

from multiprocessing import Pool


def insert_acc(date_starts_per_model):
    p = Pool()
    targets = {}
    for x in config['GRANULE']:
        targets[x] = (prices.shift(freq='-' + x) - prices).dropna().apply(np.sign)
    granule = ''
    if 'lgbm' in config['NAMES']:
        for x in config['NAMES']['lgbm']['PARAMS']:
            targets[x[0]] = (prices.shift(freq='-' + x[0]) - prices).dropna().apply(np.sign)
            granule = 'min' if x[0][-1] == 'n' else 'h'
    print('targets: ', targets.keys())
    accuracy_dict = {}
    for pr_key in predictions_dict.keys():
        db_key = pr_key.split('_')[-1]
        try:
            date_st = date_starts_per_model[db_key]
            if date_st == 'collector':
                continue
            date_st = int(pd.Timestamp(date_st, unit='s').timestamp())
        except:
            date_st = 0
        pr = pd.DataFrame(client.execute('''select ts, price_del
         from db_binance.tbl_{pipe_id}_{db_key}_logs where ts >= {date_st} order by ts asc'''\
                                         .format(pipe_id=pipe_id,
                                                       db_key=db_key,
                                                       date_st=date_st-(10*24*60*60))))
        pr.columns = ['ts', 'price_del']
        pr.index = pd.to_datetime(pr['ts'], unit='s')
        pr = pr.drop(['ts'], axis=1)
        name = pr_key.split('_')[0]
        number = pr_key.split('_')[-1]
        if name == 'ld':
            continue
        if name == 'lgbm' or name == 'american|class' or name == 'rnn|class':
            target_key = number[1:3]
            if target_key[0] == '0':
                target_key = target_key[1]

            target_key = target_key + granule
            target = targets[target_key]
        else:
            target_key = config['GRANULE'][0]
            target = targets[target_key]
        test_df = target.join(pr.apply(np.sign), how='inner')
        print(test_df.head())
        test_df.columns = ['target', 'y_pred']
        test_df['ts'] = test_df.index

        inds = list(zip(np.arange(0, test_df['ts'].shape[0], 100),
                        np.arange(0, test_df['ts'].shape[0], 100)[1:]))

        if test_df['ts'].shape[0] > inds[-1][1]:
            inds.append((inds[-1][1], test_df['ts'].shape[0]))

        # inds = inds[:30]
        res_1 = []
        for x in tqdm(inds):
            params = test_df.iloc[x[0]:x[1], :].apply(lambda x: receive(x, test_df), axis=1).tolist()
            res_ = p.map(get_target_and_accuracy, params)
            res_1.append(res_)
        import itertools
        new_res_1 = list(itertools.chain(*res_1))

        new_res_1 = pd.DataFrame(new_res_1)
        new_res_1.index = test_df.iloc[:new_res_1.shape[0], :].index
        new_res_1['ts'] = new_res_1.index
        new_res_1['ts'] = new_res_1['ts'].apply(lambda x: int(pd.Timestamp(x).timestamp()))
        try:
            date_st = date_starts_per_model[db_key]
            new_res_1 = new_res_1[date_st:]
        except:
            pass
        accuracy_dict[pr_key] = new_res_1

/*
------------- UNDER NDA -----------------------
*/

        create_query = create_query.format(exchange='binance', pipe_name=pipe_id, pipe_model=number)

        client.execute(create_query.format(exchange='binance', pipe_name=pipe_id, pipe_model=number))

        print(create_query)

        acc_query = acc_query.format(exchange='binance', pipe_name=pipe_id, pipe_model=number)

        print(acc_query)

        client.execute(acc_query, list_to_insert)

    print('acc success')


def define_val_mode():
    if int(pd.Timestamp(config['DATE_END_TEST']).timestamp()) > \
            int(pd.Timestamp(config['DATE_END_TRAIN']).timestamp()):
        usual_val = True
    else:
        usual_val = False
    return usual_val


def define_hole_mode(m_key):
    check_if_table_exists = m_key
    print("M_key: ", m_key)
    select_q = '''select ts, price, price_del 
    from db_binance.tbl_{pipe_id}_{check_if_table_exists}_logs order by ts asc''' \
        .format(pipe_id=pipe_id,
                check_if_table_exists=check_if_table_exists)
    try:
        table_exits = pd.DataFrame(client.execute(select_q))
        if table_exits.shape[0] != 0:
            hole_mode = True
        else:
            hole_mode = False
    except:
        hole_mode = False
    return hole_mode


def define_max_start_end(hole_mode, usual_val, m_keys):
    if hole_mode:
        select_q = '''select ts, price, price_del 
        from db_binance.tbl_{pipe_id}_{model_id}_logs where ts >= {start} and ts <= {finish} order by ts asc'''
        date_end_test = str(pd.Timestamp.utcnow().round('h').replace(tzinfo=None) - pd.Timedelta('30min'))
        date_end_test_ts = int(pd.Timestamp(date_end_test).timestamp())
        if usual_val:
            date_start_test_ts = int(pd.Timestamp(config['DATE_START_TEST']).timestamp())
        else:
            date_start_test_ts = int((pd.Timestamp(config['DATE_END_TRAIN']) - pd.Timedelta('30d')).timestamp())

        date_starts = {}
        date_starts_li = []
        for x in m_keys:
            filled_select_q = select_q.format(pipe_id=pipe_id,
                                              model_id=x,
                                              start=date_start_test_ts,
                                              finish=date_end_test_ts)
            logs = pd.DataFrame(client.execute(filled_select_q))
            if logs.shape[0] == 0:
                last_ts = date_start_test_ts
                is_collector = False
            else:
                logs.columns = ['ts', 'price', 'price_del']
                last_ts = logs.tail(1)['ts']
                collector_sign = int(str(pd.Timestamp(int(last_ts), unit='s')).split(' ')[1].split(':')[-1][-1])
                if collector_sign == 8:
                    is_collector = True
                else:
                    is_collector = False
            if not is_collector:
                date_starts[x] = str(pd.Timestamp(int(last_ts), unit='s'))
                date_starts_li.append(int(last_ts))
            else:
                date_starts[x] = 'collector'
        print(date_starts_li)
        date_start_ts = min(date_starts_li)
        date_start_str = str(pd.Timestamp(date_start_ts, unit='s') - pd.Timedelta('30d'))
        date_end_test_str = str(pd.Timestamp(date_end_test_ts, unit='s'))

        return date_start_str, date_end_test_str, date_starts
    else:
        if usual_val:
            date_end_test = str(pd.Timestamp.utcnow().round('h').replace(tzinfo=None) - pd.Timedelta('1h'))
            
            date_start_test = str(pd.Timestamp(config['DATE_START_TEST']) - pd.Timedelta('30d'))
        else:
            date_start_test = str(pd.Timestamp(config['DATE_START_TEST']) - pd.Timedelta('30d'))
            date_end_test = str(pd.Timestamp(config['DATE_END_TEST']))
        return date_start_test, date_end_test, {}


models_list = config['MODELS_LIST']
print("Models_list: ", models_list)
val_mode = define_val_mode()
hole_mode = define_hole_mode(models_list[0])
date_start_test, date_end_test, date_starts_per_model = define_max_start_end(hole_mode, val_mode, models_list)
print("Models_list: ", models_list, "\n",
      "Val mode: ", val_mode, "\n",
      "Hole_mode: ", hole_mode, "\n",
      "Date_start_test: ", date_start_test, '\n',
      "Date_end_test: ", date_end_test, "\n",
      "Date_starts_per_model: ", date_starts_per_model)

if int(pd.Timestamp(date_start_test).timestamp()) > int(pd.Timestamp('2019-09-01').timestamp()):
    config['DEALINGS_PORT'] = binance_port
    config['DEALINGS_SERVER'] = binance_server
    config['DEALINGS_USERNAME'] = binance_username
    config['DEALINGS_PASSWORD'] = binance_password
    config['EXCHANGE'] = 'BINANCE'
    config['QUOTE'] = binance_quote

    
scaler = pickle.load(open('./models/data_scaler_1.sav', 'rb'))

models_tbls = {
    'ld': '0',
    'knn': '1',
    'knn_pca': '2',
    'rnn': '3',
    'american_v4': '4',
    'lgbm': '5'
}

names = init_names()
print('names: ', names)
models_dict = load_models()
print('models_dict: ', models_dict.keys())

if isinstance(config['BASE'], str):

    dsb = DatasetBuilder(config, date_start_test, date_end_test, new_five=True)
    features, prices = dsb.get_train_dataset(return_price=True)
    prices = pd.DataFrame(prices)
    prices.columns = ['price']

else:
    config_ = config.copy()
    config_['BASE'] = config['BASE'][0]
    config_['IND_FRAC'] = config['IND_FRAC_' + config['BASE'][0]]
    dsb = DatasetBuilder(config_, date_start_test, date_end_test, new_five=True)
    features, prices = dsb.get_train_dataset(return_price=True)
    prices = pd.DataFrame(prices)
    prices.columns = ['price']

    for base in config['BASE'][1:]:
        config_ = config.copy()
        config_['BASE'] = base
        config_['IND_FRAC'] = config['IND_FRAC_' + base]
        config_['USE_STAKAN_FEATURES'] = False
        dsb = DatasetBuilder(config_, date_start_test, date_end_test, new_five=True)
        features_etc, _ = dsb.get_train_dataset(return_price=True)
        features_etc = features_etc[['normed_dsma_89_89_1', 'bbe_78_2_1', 'bbe_97_97_1']]
        features = features.join(features_etc, rsuffix='_' + base)

features = features.dropna()


if 'target' in features.columns:
    features = features.drop(['target'], axis=1)
features = features[date_start_test:date_end_test]
features.loc[:] = scaler.transform(features)

# features.to_csv('features.csv')
features_dict = get_features_for_models()
print('features_dict: ', features_dict.keys())
predictions_dict = get_predictions(models_dict, features_dict, date_starts_per_model)
print('predictions_dict: ', predictions_dict.keys())
insert_all_in_db(predictions_dict)
insert_acc(date_starts_per_model)

if config.get('USE_MMI_FEATURES', None):
    config['MAIN_BASE'] = config['BASE']
    config['BASE'] = 'MMI'
    config['MMI_MODE'] = True
    json.dump(config, open(config['PATH_CONFIG'], 'w'),
              sort_keys=True, indent=4, separators=(',', ': '))

if config['STRATEGY_PIPELINE']:
    strategy_url = 'http://192.168.88.52:5001/start'
    headers = {'Content-type': 'application/json'}
    models_list = [pipe_id + '_' + m for m in config['MODELS_LIST'] if m!='1']
    learn_rate = np.round(np.percentile((client.execute('select price_del from db_binance.tbl_{pipe_id}_0_logs order by ts desc'.format(pipe_id=pipe_id))), 95), 1)
    models_data = {
        "model_names": models_list,
        "experiment_name": "base",
        "learn_rate": learn_rate
    }
    
    r = requests.post(url=strategy_url, data=json.dumps(models_data), headers=headers)
