import pickle
from cabal_utils.layers import Attention
from cabal_utils.losses import get_corr_and_mse_loss
from keras.models import load_model

import json
import os
import keras

# os.environ['PATH_CONFIG'] = '/home/bogan/processing_oracle/config.json'
# os.environ['PATH_PIPELINE_ID'] = '/home/bogan/processing_oracle/7'

config = json.load(open(os.environ['PATH_CONFIG']))

custom_objects = {'corr_loss_and_mse': get_corr_and_mse_loss(1, 100),
                  "Attention": Attention}

# from KnnPcaPredictDataHandler import KnnPcaPredictDataHandler
from pipe_bento.KerasPredictDataHandler import KerasPredictDataHandler
from pipe_bento.KnnPredictDataHandler import KnnPredictDataHandler
from pipe_bento.KnnMeanDataHandler import KnnMeanDataHandler
from pipe_bento.KnnPcaPredictDataHandler import KnnPcaPredictDataHandler
from pipe_bento.LightGbmPredictDataHandler import LightGbmPredictDataHandler
from pipe_bento.KerasCategoricalPredictDataHandler import KerasCategoricalPredictDataHandler
from pipe_bento.KnnCategoricalPredictDataHandler import KnnCategoricalPredictDataHandler
from pipe_bento.KnnPcaCategoricalPredictDataHandler import KnnPcaCategoricalPredictDataHandler

# scaler_path = os.environ['PATH_PIPELINE_ID'] + '/' + config['PATH_MODELS'] + '/' + 'data_scaler_1.sav'
# scaler = pickle.load(open(scaler_path, 'rb'))

for key in config['NAMES']:
    if key == 'american_v4':
        model_path = os.environ['PATH_PIPELINE_ID'] + '/' + \
                     config['PATH_MODELS'] + '/' + 'american_v4' + '.h5'

        model = load_model(model_path, custom_objects=custom_objects)

        bento_model = KerasPredictDataHandler.pack(model=model)

        saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' + \
                                      config['PATH_MODELS'] + '/' + 'american_v4')

        print('american saved to bento')

    if key == 'rnn':
        model_path = os.environ['PATH_PIPELINE_ID'] + '/' + \
                     config['PATH_MODELS'] + '/' + 'rnn' + '.h5'

        model = load_model(model_path, custom_objects=custom_objects)

        bento_model = KerasPredictDataHandler.pack(model=model)

        saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' + \
                                      config['PATH_MODELS'] + '/' + 'rnn')
        print('rnn saved to bento')
    if key == 'knn':

        knn_path = os.environ['PATH_PIPELINE_ID'] + '/' + \
                   config['PATH_MODELS'] + '/' + 'knn.pkl'

        knn = pickle.load(open(knn_path, 'rb'))
        knn.n_jobs = 1

        bento_model = KnnPredictDataHandler.pack(knn=knn)

        saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' + \
                                      config['PATH_MODELS'] + '/' + 'knn')
        if config['NAMES']['knn'].get('LEARNING_DISTANCE'):
            bento_model = KnnMeanDataHandler.pack(knn=knn)

            saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' + \
                                          config['PATH_MODELS'] + '/' + 'knn_mean')

        print('knn saved to bento')

    if key == 'knn_pca':
        knn_path = os.environ['PATH_PIPELINE_ID'] + '/' + \
                   config['PATH_MODELS'] + '/' + 'knn_pca.pkl'

        knn = pickle.load(open(knn_path, 'rb'))
        knn.n_jobs = 1

        pca_path = os.environ['PATH_PIPELINE_ID'] + '/' + \
                   config['PATH_MODELS'] + '/' + 'pca.pkl'

        pca = pickle.load(open(pca_path, 'rb'))

        bento_model = KnnPcaPredictDataHandler.pack(knn=knn, pca=pca)

        saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' + \
                                      config['PATH_MODELS'] + '/' + 'knn_pca')
        print('knn_pca saved to bento')

    if key == "lgbm":
        params = config['NAMES']['lgbm']['PARAMS']
        for param in params:
            frame, percent = param

            model_path = os.environ['PATH_PIPELINE_ID'] + "/" + config['PATH_MODELS'] + \
                         '/lgbm_{frame}_{percent}.pkl'.format(frame=frame, percent=str(percent)[2:])
            scaler_path = os.environ['PATH_PIPELINE_ID'] + "/" + config['PATH_MODELS'] + "/data_scaler_1.sav"

            lgbm = pickle.load(open(model_path, 'rb'))
            scaler = pickle.load(open(scaler_path, 'rb'))

            bento_model = LightGbmPredictDataHandler.pack(scaler=scaler, mylgbm=lgbm)

            saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' +
                                          config['PATH_MODELS'] + '/' + 'lgbm_{frame}_{percent}'
                                          .format(frame=frame, percent=str(percent)[2:]))

            model_path = os.environ['PATH_PIPELINE_ID'] + "/" + config['PATH_MODELS'] + \
                        '/american|class_{frame}_{percent}.h5'.format(frame=frame, percent=str(percent)[2:])

            model = load_model(model_path, custom_objects=custom_objects)

            bento_model = KerasCategoricalPredictDataHandler.pack(scaler=scaler, model=model)

            saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' +
                                         config['PATH_MODELS'] + '/' + 'american|class_{frame}_{percent}'
                                         .format(frame=frame, percent=str(percent)[2:]))

            model_path = os.environ['PATH_PIPELINE_ID'] + "/" + config['PATH_MODELS'] + \
                         '/rnn|class_{frame}_{percent}.h5'.format(frame=frame, percent=str(percent)[2:])

            model = load_model(model_path, custom_objects=custom_objects)

            bento_model = KerasCategoricalPredictDataHandler.pack(scaler=scaler, model=model)

            saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' +
                                          config['PATH_MODELS'] + '/' + 'rnn|class_{frame}_{percent}'
                                          .format(frame=frame, percent=str(percent)[2:]))
            model_path = os.environ['PATH_PIPELINE_ID'] + "/" + config['PATH_MODELS'] + \
                         '/knn|class_{frame}_{percent}.pkl'.format(frame=frame, percent=str(percent)[2:])

            model = pickle.load(open(model_path, 'rb'))
            model.n_jobs = 1

            bento_model = KnnCategoricalPredictDataHandler.pack(scaler=scaler, knn=model)

            saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' +
                                          config['PATH_MODELS'] + '/' + 'knn|class_{frame}_{percent}'
                                          .format(frame=frame, percent=str(percent)[2:]))

            knn_path = os.environ['PATH_PIPELINE_ID'] + '/' + \
                      config['PATH_MODELS'] + '/' + 'knnpca|class_{frame}_{percent}.pkl'.format(frame=frame, percent=str(percent)[2:])

            knn = pickle.load(open(knn_path, 'rb'))
            knn.n_jobs = 1

            pca_path = os.environ['PATH_PIPELINE_ID'] + '/' + \
                      config['PATH_MODELS'] + '/' + 'pca|class_{frame}_{percent}.pkl'.format(frame=frame, percent=str(percent)[2:])

            pca = pickle.load(open(pca_path, 'rb'))

            bento_model = KnnPcaCategoricalPredictDataHandler.pack(knn=knn, pca=pca, scaler=scaler)

            saved_path = bento_model.save(os.environ['PATH_PIPELINE_ID'] + '/' + \
                                         config['PATH_MODELS'] + '/' + \
                                          'knnpca|class_{frame}_{percent}'.format(frame=frame, percent=str(percent)[2:]))

print('all models saved to bento')
