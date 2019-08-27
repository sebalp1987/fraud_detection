import datetime
import os


import pandas as pd

from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import mini_batch_kmeans
from fraud_home.resources.fraud_home.model_utils import oversample_unsupervised


def mini_batch_kmeans_tunning(normal_df: pd.DataFrame, anormal_df: pd.DataFrame, oversample_times,
                              batch_size_range=range(100, 1001, 100), max_iter=10001,
                              n_clusters=10):
    if oversample_times is None:
        normal_df, anormal_df = oversample_unsupervised(normal_df, anormal_df)
    else:
        anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    fraud_list_score = []
    i = 0
    for iteration in range(5000, max_iter, 5000):
        for batch_size_i in batch_size_range:
            for n_clusters_i in range(2, n_clusters, 1):
                i += 1
                f1, f2, fraud_score, _ = mini_batch_kmeans.mini_batch_kmeans(normal_df, anormal_df, max_iter=iteration,
                                                                             batch_size=batch_size_i,
                                                                             n_clusters=n_clusters_i)

                fraud_list_score.append([iteration, batch_size_i, n_clusters_i, f1, f2, fraud_score])
    fraud_list_score = pd.DataFrame(fraud_list_score,
                                    columns=['max_iter', 'batch_size', 'n_clusters', 'f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]

    df = pd.DataFrame(fraud_list_score, columns=['max_iter', 'batch_size', 'n_clusters', 'f1', 'f2', 'fraud_score'])
    day = datetime.datetime.today().strftime('%Y-%m-%d')
    os.makedirs(os.path.dirname(STRING.monitoring_path_no_supervisado), exist_ok=True)
    df.to_csv(STRING.monitoring_no_supervisado_kmeans, sep=';', encoding='latin1', index=False)

    return max_fraud_score
