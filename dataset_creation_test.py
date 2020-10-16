import json
import os
import warnings

import numpy as np
import pandas as pd
from oracle_data.DataLoader import DatasetBuilder
from clickhouse_driver import Client

from bch_features import get_blockchain_features

# os.environ['PATH_CONFIG'] = '/home/bogan/processing_oracle/config.json'
# os.environ['PATH_PIPELINE_ID'] = '/home/bogan/processing_oracle/7'

warnings.filterwarnings("ignore")

pd.set_option('precision', 10)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

config = json.load(open(os.environ['PATH_CONFIG']))
config['PATH_PIPELINE_ID'] = os.environ['PATH_PIPELINE_ID']
json.dump(config, open(os.environ['PATH_CONFIG'], 'w'), sort_keys=True, indent=4, separators=(',', ': '))
config = json.load(open(os.environ['PATH_CONFIG']))
# start = pd.to_datetime(config['DATE_START'])
# end = pd.to_datetime(config['DATE_END'])

start = config['DATE_START']
end = config['DATE_END']

if config.get('USE_LICV_FEATURES', None):
    binance_username = 'default'
    binance_server = 'host'
    binance_port = 9000
    binance_quote = 'USDT'
    binance_password = 'password'

    client = Client(user=binance_username,
                    host=binance_server,
                    port=binance_port,
                    password=binance_password)

    def load_licv():
        licv_long = client.execute('''select ts, price, origQty, executedQty, averagePrice
        from db_BTC_USDT.AllForceOrders_buy
        order by ts ASC''')
        import pandas as pd
        licv_long = pd.DataFrame(licv_long)
        licv_long.columns = ['ts', 'price', 'origQty', 'executedQty', 'averagePrice']
        licv_long.index = pd.to_datetime(licv_long['ts'], unit='ms')

        licv_short = client.execute('''select ts, price, origQty, executedQty, averagePrice
        from db_BTC_USDT.AllForceOrders_sell
        order by ts ASC''')
        import pandas as pd
        licv_short = pd.DataFrame(licv_short)
        licv_short.columns = ['ts', 'price', 'origQty', 'executedQty', 'averagePrice']
        licv_short.index = pd.to_datetime(licv_short['ts'], unit='ms')
        licv_short['origQty'] *= -1
        licv_short['executedQty'] *= -1
        licv = pd.concat((licv_long, licv_short), axis=0).sort_index()
        licv = licv.drop(columns=['ts'], axis=1)

        group = licv.groupby(pd.Grouper(freq='1min'))['executedQty'].agg({'sum': sum, 'count': len})
        # g['diff_count'] =g['count'].diff().fillna(0)
        first_price = licv.groupby(pd.Grouper(freq='1min'))['averagePrice'].nth([0])
        last_price = licv.groupby(pd.Grouper(freq='1min'))['averagePrice'].nth([-1])
        first_last_price = pd.DataFrame({'first_price': first_price, 'last_price': last_price})
        first_last_price = first_last_price.resample('1min').mean().fillna(method='ffill').fillna(0)
        # change[change == 0] = 0.1
        # g['lic_f4'] = g['sum'] / change
        return group, first_last_price


    def licv_window(features, first_last_price, window):
        f1 = np.around(features.rolling(window)['sum'].sum(), decimals=7)
        f2 = np.around(features.rolling(window)['count'].sum(), decimals=7)
        f3 = f2.diff().fillna(0)
        if window == '1min':
            change = first_last_price.apply(lambda x: (((max(x) / min(x)) * 100) - 100), axis=1)
        else:
            change = (first_last_price.mean(axis=1).rolling(window).apply(lambda x:
                                                                          (max(x[0], x[-1])) /
                                                                          (min(x[0], x[-1]))) * 100) - 100

        change[change == 0] = 0.1
        f4 = f1 / change.rolling(window).sum()

        ret = pd.DataFrame({'licv_f1': f1,
                            'licv_f2': f2,
                            'licv_f3': f3,
                            'licv_f4': f4})
        ret.columns = [col + '_' + window for col in ret.columns]
        return ret
 /*
------------- UNDER NDA -----------------------
*/




    if config.get('USE_RNG_FEATURES', False):
        datasets_dir = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_DATASETS'] + '/'
        all_ = pd.read_csv(datasets_dir + config['TRAIN_DATASET'], parse_dates=True, index_col=[0])

        noise_1 = np.random.uniform(-3, 3, all_.shape[0])
        noise_2 = np.random.normal(-3, 3, all_.shape[0])
        noise_3 = np.random.uniform(-3, 3, all_.shape[0])
        noise_4 = np.random.normal(-3, 3, all_.shape[0])
        noise_5 = np.random.uniform(-3, 3, all_.shape[0])

        all_['rng_1'] = noise_1
        all_['rng_2'] = noise_2
        all_['rng_3'] = noise_3
        all_['rng_4'] = noise_4
        all_['rng_5'] = noise_5

        print('I AM IN RANDOM FEATURES SECTION')
        print('Join random features')
        print(all_.filter(like='rng').columns)
        all_.to_csv(datasets_dir + config['TRAIN_DATASET'])

    if 'USE_BCH_FEATURES' in config:
        if config['USE_BCH_FEATURES']:
            datasets_dir = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_DATASETS'] + '/'
            all_ = pd.read_csv(datasets_dir + config['TRAIN_DATASET'], parse_dates=True, index_col=[0])
            bch = get_blockchain_features()
            all_ = all_.join(bch).dropna()
            print('I AM IN BLOCKCHAIN FEATURES SECTION')
            print('Join bch')
            print(all_.filter(like='bf').columns)
            all_.to_csv(datasets_dir + config['TRAIN_DATASET'])

