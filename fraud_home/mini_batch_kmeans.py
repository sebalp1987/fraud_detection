import pandas as pd
from sklearn.cluster import MiniBatchKMeans

import fraud_home.resources.fraud_home.fraud_score as fs


def mini_batch_kmeans(normal, anormal, n_clusters=2, max_iter=100, batch_size=100):
    x = pd.concat([normal, anormal], axis=0)
    x_fraude = x[['id_siniestro', 'FRAUDE']]
    del x['FRAUDE']
    del x['id_siniestro']

    db = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter, batch_size=batch_size, random_state=42)
    db.fit(x)
    labels = db.predict(x)
    labels_df = pd.DataFrame(labels, index=x.index, columns=['Clusters'])

    comparative = pd.concat([x_fraude, labels_df], axis=1)
    f1, f2, fscore, df_clusters = fs.fraud_score(comparative.drop(['id_siniestro'], axis=1), 'FRAUDE', 'Clusters')
    comparative['FRAUDE_Clusters'] = pd.Series(0, index=comparative.index)
    comparative['FRAUDE'] = comparative['FRAUDE'].map(int)
    comparative.loc[comparative['FRAUDE'] == 1, 'FRAUDE_Clusters'] = 1
    comparative.loc[comparative['Clusters'].isin(df_clusters), 'FRAUDE_Clusters'] = 1
    return f1, f2, fscore, comparative
