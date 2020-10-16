# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import pyodbc 
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import json
import os
import pickle

pd.set_option('precision', 10)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)


# In[2]:


config = json.load(open(os.environ['PATH_CONFIG']))


# In[3]:


def get_dealings_from_sql(
/*
------------- UNDER NDA -----------------------
*/

    return dealings


# In[4]:


start = pd.Timestamp(config['DATE_START']) 
start = (start - pd.Timedelta('5d'))

finish = pd.Timestamp(config['DATE_END'])

print('Fetching dealings from', start.date(),'to', finish.date(), '...')
dealings = get_dealings_from_csv(start=start.timestamp(), finish=finish.timestamp())
print('...done.')


# In[5]:


def get_groupped_dealings(dealings): 
    groupped_dealings = pd.DataFrame()
    groups = dealings.groupby('_30m')
    groupped_dealings['price_sum'] = groups['price'].sum()
    groupped_dealings['qty'] = groups['qty'].sum()
    groupped_dealings['amount'] = groups['_timestamp'].count()
    groupped_dealings['mean_price'] = groups['price'].mean()
    groupped_dealings['cost_sum'] = groups['cost'].sum()
    groupped_dealings['avg_price'] = groupped_dealings['cost_sum'] / groupped_dealings['qty']
        
    groupped_dealings['open_price'] = groups[['price']].first()
    groupped_dealings['close_price'] = groups[['price']].last()
    groupped_dealings['max_price'] = groups[['price']].max()
    groupped_dealings['min_price'] = groups[['price']].min()
    
    groupped_dealings['sell_qty'] =             dealings[dealings['type'] == 'SELL'].groupby('_30m')['qty'].sum()
    
    groupped_dealings['buy_qty'] =             dealings[dealings['type'] == 'BUY'].groupby('_30m')['qty'].sum()
    return groupped_dealings
    


# In[6]:


def feature1_for_n_intervanls(groupped_dealings,window_size):
    window_sum = groupped_dealings.rolling(
                        window = window_size, min_periods=1).sum()
    avg_price_1 = window_sum['cost_sum']/ window_sum['qty']
    avg_price_2 = avg_price_1.shift(window_size)
    return (avg_price_1 - avg_price_2)/avg_price_2

def feature2_for_n_intervanls(groupped_dealings,window_size):
    window_sum = groupped_dealings.rolling(
                        window = window_size, min_periods=1).sum()
    
    mean_price_1 = window_sum['price_sum']/ window_sum['amount']
    mean_price_2 = mean_price_1.shift(window_size)
    return (mean_price_1 - mean_price_2)/mean_price_2

def feature3_for_n_intervanls(groupped_dealings,window_size):
    
    open_price = groupped_dealings['open_price'].shift(window_size-1)
    close_price = groupped_dealings['close_price']
    
    return (close_price - open_price)/open_price

def feature4_for_n_intervanls(groupped_dealings, window_size):
    rolling_groupped_dealings = groupped_dealings.rolling(
                            window = window_size, min_periods=1)
    winndow_min_price = rolling_groupped_dealings['min_price'].min()
    winndow_max_price = rolling_groupped_dealings['max_price'].max()

    return (winndow_max_price - winndow_min_price)/winndow_min_price

def feature5_for_n_intervanls(groupped_dealings, window_size):
    rolling_groupped_dealings = groupped_dealings.rolling(
                            window = window_size, min_periods=1)
    
    sell_dealings_qty_sum = rolling_groupped_dealings['sell_qty'].sum()
    buy_dealings_qty_sum = rolling_groupped_dealings['buy_qty'].sum()

    feature5 = (buy_dealings_qty_sum - sell_dealings_qty_sum)/(buy_dealings_qty_sum + sell_dealings_qty_sum)
    return np.tanh(feature5)/np.tanh(1)

def feature6_for_n_intervanls(groupped_dealings, window_size):
    qty = groupped_dealings['qty'].rolling(window_size).sum()
    max_qty = qty.rolling(100).max()
    ans = qty/max_qty
    return ans

def target(groupped_dealings):
    
    next_weighted_price = groupped_dealings['avg_price'].shift(-1)
    close_price = groupped_dealings['close_price']
    
    return (next_weighted_price - close_price)/close_price


def get_old_features(groupped_dealings):
    dealings = groupped_dealings
    
    data = pd.DataFrame()
/*
------------- UNDER NDA -----------------------
*/
    
    return data


# In[7]:


def ema(groupped_dealings, n=5):
    emas = pd.DataFrame()
    emas['ema_' + str(n)] = groupped_dealings['close_price'].ewm(span=n).mean()
    return emas
    

def bbe(groupped_dealings, n=10, d=5):
    
    emas = ema(groupped_dealings, n=n)
    bbes = pd.DataFrame()
    stds = pd.DataFrame()
    stds['ema_std_' + str(n)] = groupped_dealings['close_price'].ewm(span=n).std()
    
    
    # bbe_1 - это ема плюс ско (линия выше), bbe_2 линия ниже соответственно
    
    bbes['bbe_' + str(n) + '_' + str(d) + '_' + '1'] = emas['ema_' + str(n)] + d * stds['ema_std_' + str(n)]
    bbes['bbe_' + str(n) + '_' + str(d) + '_' + '2'] = emas['ema_' + str(n)] - d * stds['ema_std_' + str(n)]
    
    
    return bbes 

def dema(dealings, n_1=5, n_2=5):
    emas = ema(dealings,  n_1)
    
    ema_ema = pd.DataFrame()
    ema_ema['ema_' + str(n_2)] = emas['ema_' + str(n_1)].ewm(span=n_2).mean()
    
    
    dema = pd.DataFrame()
    
    dema['dema_' + str(n_1) + '_' + str(n_2)] = 2 * emas['ema_' + str(n_1)] - ema_ema['ema_' + str(n_2)]
    
    
    return dema
    

from ta import vortex_indicator_pos, vortex_indicator_neg
def vortex(groupped_dealings, n=10):
    vortex = pd.DataFrame()
    vortex['vortex_' + str(n) + '_1'] = vortex_indicator_pos(high = groupped_dealings['close_price'],                      low = groupped_dealings['min_price'],                     close = groupped_dealings['close_price'],                      n=n, fillna=False )

    vortex['vortex_' + str(n) + '_2'] = vortex_indicator_neg(high = groupped_dealings['close_price'],                      low = groupped_dealings['min_price'],                     close = groupped_dealings['close_price'],                      n=n,fillna=False )
    return vortex


def dsma(dealings, n_1=5, n_2=10):
    dsma = pd.DataFrame()
    
    dsma['dsma_' + str(n_1) + '_' + str(n_2) + '_1'] = dealings['close_price'].rolling(n_1).mean()
    dsma['dsma_' + str(n_1) + '_' + str(n_2) + '_2'] = dealings['close_price'].rolling(n_2).mean()
    
    return dsma 

def cci(dealings, n=5):
    cci = pd.DataFrame()
    tp = (dealings['max_price'] + dealings['min_price'] + dealings['close_price']) / 3
    k = ( 1/ 0.015 )
    
    cci['cci_' + str(n)] = k * (tp - tp.rolling(n).mean() / tp.rolling(n).std())
    
    return cci


# In[8]:


def make_dataset(dealings):
/*
------------- UNDER NDA -----------------------
*/
    
    return data


# In[9]:


def make_ith_dataset(dealings_, i):
/*
------------- UNDER NDA -----------------------
*/
    return dataset

print('Creating dataset...')
data = pd.concat([
    make_ith_dataset(dealings, i) for i in tqdm(range(6))])
data.index = pd.to_datetime(data.index*5, unit='m')
data = data.loc[~data.index.duplicated(keep='first')]
print('...done.')


# In[10]:


def normalize(df_):
    df = df_.copy()
    close = df[['close']]
    #bbe
    bbe = df.filter(like='bbe')
    df[bbe.columns] = np.abs(bbe.values - close.values) / bbe.values
#     bbe_pair = np.array_split(bbe.columns, len(bbe.columns) / 2)
#     for pair in bbe_pair:
#         higher = (df[pair[0]] > df[pair[1]]).astype(int).diff().fillna(0)
#         onehotencoder = OneHotEncoder(categorical_features = [0])
#         higher = onehotencoder.fit_transform(higher.values.reshape(-1, 1) + 1).toarray()

#         s = pd.DataFrame(higher)
#         ind = pair[0].split('_')
#         pref = 'bbe_' + ind[1] + '_' + ind[2] + '_' 
#         s.columns = [pref + 'change_down', pref + 'none', pref + 'change_up']
#         s.index = data.index
#         df = pd.concat((df, s), axis=1)
    #rsi
    rsi = df.filter(like='rsi')
    df[rsi.columns] = rsi.values/100.
    #dema
    dema = df.filter(like='dema')
    df[dema.columns] = (dema.values - close.values) / dema.values
    return df

def norm_dsma(data):
    all_cols = data.columns.str.lower() 
    dsma = list(all_cols[['dsma' in x for x in all_cols]])
    
    def norm_dsma_pair(x): 
        cols = list(x)
        dsma_df = data[cols]
        normed = (dsma_df.iloc[:,0].values - dsma_df.iloc[:,1].values) /              (dsma_df.iloc[:,0].values + dsma_df.iloc[:,1].values)

        inds = 'normed_dsma_' + cols[0].split('_')[1] + '_' + cols[0].split('_')[2]
        normed_index = dsma_df.index
        normed = pd.Series(normed)
        normed.index = normed_index
        data[inds] = normed
        return data
    
    zip_dsma = list(zip(dsma[0:len(dsma)], dsma[1:]))[::2]    
    [norm_dsma_pair(x) for x in zip_dsma]    
    return data

def norm_vortex(data):    
    all_cols = data.columns.str.lower()     
    dsma = list(all_cols[['vortex' in x for x in all_cols]])    
    def norm_vortex_pair(x):        
        cols = list(x)
        dsma_df = data[cols]
        normed =         (dsma_df.iloc[:,0].values - dsma_df.iloc[:,1].values) /              (dsma_df.iloc[:,0].values + dsma_df.iloc[:,1].values)
        normed_index = dsma_df.index
        normed = pd.Series(normed)
        normed.index = normed_index
        inds = 'normed_vortex_' +             cols[0].split('_')[1]         
        data[inds] = normed
        return data
    
    zip_dsma = list(zip(dsma[0:len(dsma)], dsma[1:]))[::2]
    [norm_vortex_pair(x) for x in zip_dsma]
    return data


def nomalize_data(data_):
    data = data_.copy()
    data = normalize(data)
    data = norm_dsma(data)
    data = norm_vortex(data)
    return data


print('Normalizind data...')
normed_data = nomalize_data(data).dropna()
print('...done.')
print('Normed data shape', normed_data.shape)


normed_features = pd.concat([
    normed_data.filter(like='feature'),
    normed_data.filter(like='normed'),
    normed_data.filter(like='dema'),
    normed_data.filter(like='bbe'),
    normed_data.filter(like='rsi')], axis=1)

# display(normed_features.columns)

target = normed_data['target']

X_train = normed_features[config['DATE_START_TRAIN']
                            :config['DATE_END_TRAIN']]
X_test = normed_features[config['DATE_START_TEST']
                               :config['DATE_END_TEST']]

y_train = target[config['DATE_START_TRAIN']
                            :config['DATE_END_TRAIN']]
y_test = target[config['DATE_START_TEST']
                               :config['DATE_END_TEST']]

X_train = pd.concat([X_train[start:finish] for start, finish in config['TORN_TRAIN']])
y_train = pd.concat([y_train[start:finish] for start, finish in config['TORN_TRAIN']])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train.loc[:] = scaler.fit_transform(X_train)
X_test.loc[:] = scaler.transform(X_test)

scalerfile = os.environ['PATH_PIPELINE_ID']+'/' + config['PATH_MODELS'] + '/' + 'data_scaler.sav'

pickle.dump(scaler, open(scalerfile, 'wb'))

train_dataset = pd.concat([X_train, y_train], axis=1)
test_dataset = pd.concat([X_test, y_test], axis=1)

def validate_dataset(data):
    features = data.drop(['target'], axis=1)
    target = data['target']
    corr = features.corrwith(target)
    
    if corr.max() > config['SUSPICIOUS_CORRELATION']:
        print('Correlation: ', corr.abs().sort_values(ascending=False).head())
        raise ValueError("Suspicious correlation")
    
print('Data validation...')
validate_dataset(pd.concat([train_dataset, test_dataset]))
print('...done.')

datasets_dir = os.environ['PATH_PIPELINE_ID']+'/'+config['PATH_DATASETS']+'/'
train_dataset.to_csv(datasets_dir + config['TRAIN_DATASET'])
test_dataset.to_csv(datasets_dir + config['TEST_DATASET'])

print('Datasets saved.')
