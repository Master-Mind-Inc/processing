import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle
import json
from clickhouse_driver import Client
#
# os.environ['PATH_CONFIG'] = 'config.json'
# os.environ['PATH_PIPELINE_ID'] = '7'

config = json.load(open(os.environ['PATH_CONFIG']))
#
# client = Client(password=os.environ["DB_LOGS_PASS"],
#                 host=os.environ["DB_LOGS_HOST"],
#                 user=os.environ["DB_LOGS_USER"],
#                 port=int(os.environ["DB_LOGS_PORT"]))
client = Client(password="password",
                host="host",
                user="default",
                port=int(29000))



def scale(train, test, config):
    target_train = train['target']
    target_test = test['target']
    train = train.drop(['target'], axis=1)
    test = test.drop(['target'], axis=1)

    scaler = StandardScaler()
    train.loc[:] = scaler.fit_transform(train)
    test.loc[:] = scaler.transform(test)

    scalerfile = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'data_scaler_1.sav'
    pickle.dump(scaler, open(scalerfile, 'wb'))

    train = pd.concat((train, target_train), axis=1)
    test = pd.concat((test, target_test), axis=1)

    return train, test


def load_prices(config):
    datasets_dir = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_DATASETS'] + '/'
    price = pd.read_csv(datasets_dir + '/' + 'prices.csv', index_col=[0], parse_dates=True, header=None)
    price.columns = ['close_price']
    price['ts'] = price.index
    return price


def slice_for_models(config):
    datasets_dir = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_DATASETS'] + '/'
    all_ = pd.read_csv(datasets_dir + config['TRAIN_DATASET'], parse_dates=True, index_col=[0])
    all_ = all_.replace([np.inf, -np.inf], 0)

    if config['IS_TORN_TRAIN'] == "0":
        exp_train = config['EXPAND_TRAIN']
        exp_test = config['EXPAND_TEST']
        # exp_train = [(exp_train[0], x) for x in exp_train[1:]]
        # exp_test = [(exp_test[i], exp_test[i + 1]) for i in range(len(exp_test) - 1)]

        exp_data = []

        for x in range(len(exp_train)):
            train = all_[exp_train[x][0]:exp_train[x][1]]
            test = all_[exp_test[x][0]:exp_test[x][1]]

            train, test = scale(train, test, config)
            exp_data.append((train, test))

        return exp_data

    else:
        train = all_[config['DATE_START_TRAIN']:config['DATE_END_TRAIN']]
        test = all_[config['DATE_START_TEST']:config['DATE_END_TEST']]

        print(train.head())
        print(test.head())

        train = pd.concat([train[start:finish] for start, finish in config['TORN_TRAIN']])

        train, test = scale(train, test, config)

        exp_data = [(train, test)]

        return exp_data


def insert_predicitions_to_base(config, list_of_preds, model_name):

    datasets_dir = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_DATASETS'] + '/'

    prices = pd.read_csv(datasets_dir + 'prices.csv', index_col=False, header=None, parse_dates=True)
    prices.columns = ['_30m', 'price']

    preds = pd.concat(x for x in list_of_preds)
    preds['_30m'] = preds['_30m'].apply(lambda x: int(pd.Timestamp(x).timestamp()))
    prices['_30m'] = prices['_30m'].apply(lambda x: int(pd.Timestamp(x).timestamp()))
    preds = preds.merge(prices, on='_30m')

    preds_to_base = pd.DataFrame({'ts': preds['_30m'], 'price': preds['price'], 'price_del': preds['y_pred']})

    create_db_query = open('sql/create_db.sql').read()

    id_of_pipe = os.environ['PATH_PIPELINE_ID'].split('/')[-1].split('_')[0]

    create_db_query = create_db_query.format(pipe_name=id_of_pipe, pipe_model=model_name)

    print(create_db_query)

    create_logs_query = open('sql/create_logs.sql').read()

    create_logs_query = create_logs_query.format(pipe_name=id_of_pipe, pipe_model=model_name)

    print(create_logs_query)

    client.execute(create_db_query)

    client.execute(create_logs_query)

    insert_query = open('sql/insert.sql').read()
    insert_query = insert_query.format(pipe_name=id_of_pipe, pipe_model=model_name)

    client.execute(insert_query, list(zip(preds_to_base['ts'].values.tolist(),
                                          preds_to_base['price'].values.tolist(),
                                          preds_to_base['price_del'].values.tolist())))

    print("INSERTED PREDICTIONS TO BASE: ", os.environ['PATH_PIPELINE_ID'], " ",  model_name)


