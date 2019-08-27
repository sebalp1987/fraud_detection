import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import AllKNN

from fraud_home.configs import config

def over_sampling(x_train, y_train, model='ADASYN', ratio='minority'):
    """
    It generate synthetic sampling for the minority class using the model specificed. Always it has
    to be applied to the training set.
    :param x_train: X training set.
    :param y_train: Y training set.
    :param model: 'ADASYN' or 'SMOTE'
    :param neighbors: number of nearest neighbours to used to construct synthetic samples.
    :param ratio
    :return: xTrain and yTrain oversampled
    """
    neighbors = config.parameters.get("neighbors")
    x_train_names = x_train.columns.values.tolist()
    y_train_names = y_train.columns.values.tolist()

    if model == 'ADASYN':
        model = ADASYN(random_state=42, ratio=ratio, n_neighbors=neighbors)

    if model == 'SMOTE':
        model = SMOTE(random_state=42, ratio=ratio, k_neighbors=neighbors, m_neighbors='svm')

    x_train, y_train = model.fit_sample(x_train, y_train)

    x_train = pd.DataFrame(x_train, columns=[x_train_names])
    y_train = pd.DataFrame(y_train, columns=[y_train_names])

    return x_train, y_train


def under_sampling(x_train, y_train, neighbors=200):
    """
    It reduces the sample size for the majority class using the model specificed. Always it has
    to be applied to the training set.
    :param x_train: X training set.
    :param y_train: Y training set.
    :param neighbors: size of the neighbourhood to consider to compute the
        average distance to the minority point samples
    :return: xTrain and yTrain oversampled
    """

    x_train_names = x_train.columns.values.tolist()
    y_train_names = y_train.columns.values.tolist()

    model = AllKNN(random_state=42, ratio='majority', n_neighbors=neighbors)

    x_train, y_train = model.fit_sample(x_train_names, y_train_names)

    x_train = pd.DataFrame(x_train, columns=[x_train_names])
    y_train = pd.DataFrame(y_train, columns=[y_train_names])

    return x_train, y_train


def oversample_unsupervised(normal, anomaly):

    x = pd.concat([normal, anomaly], axis=0).reset_index(drop=True)
    x = x.copy()
    y_fraude = x[['FRAUDE']]
    id_claims = x[['id_siniestro']]
    del x['id_siniestro']
    del x['FRAUDE']
    print(x)
    print(y_fraude)
    x_train, y_train = over_sampling(x, y_fraude, model='SMOTE')
    columns_rename = ['FRAUDE'] + x_train.columns.values.tolist() + ['id_siniestro']
    y_train = pd.concat([y_train, x_train, id_claims], axis=1)
    y_train['id_siniestro'] = y_train['id_siniestro'].fillna(-1)
    y_train.columns = columns_rename

    normal = y_train[y_train['FRAUDE'] == 0]
    anomaly = y_train[y_train['FRAUDE'] == 1]

    return normal, anomaly
