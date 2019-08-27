import os
import pandas as pd
import numpy as np

from sklearn import ensemble, model_selection, metrics
from sklearn.externals import joblib

from fraud_home.resources.fraud_home import process_utils, model_utils, model_evaluation as uns, mini_batch_kmeans, \
    train_test_utils, STRING
from fraud_home.configs.config import parameters as par

# PARAMETERS
del_reduce_var = par.get('del_reduce_var')
with_feedback = par.get('with_feedback')
reduce_sample = par.get('reduce_sample')
oversample_times = par.get('oversample_times')
n_estimators = par.get("n_estimators")
max_depth = par.get("max_depth")
oob_score = par.get("oob_score")
base_sampling = par.get("base_sampling")
control_sampling = par.get("control_sampling")
bootstrap = par.get("bootstrap")
threshold_models = par.get("threshold_models")
beta = par.get("beta")
date_threshold = par.get("init_reduce_sample_date")
cp = par.get("cp")
fecha_var = par.get("fecha_var")
np.random.seed(531)


df = pd.read_csv(STRING.etl_mensual,
                 sep=';', encoding='latin1')
if with_feedback:
    open_cases = pd.read_csv(
        STRING.etl_diaria,
        sep=';', encoding='latin1')
    feedback = pd.read_csv(STRING.feedback, sep=';', encoding='latin1')
    feedback = feedback.rename(columns={'Nº SINIESTRO': 'id_siniestro', 'RESULTADO': 'resultado'})
    try:
        feedback = feedback[['id_siniestro', 'resultado']]
    except KeyError:
        feedback = feedback.rename(columns={feedback.columns.values[0]: 'id_siniestro', feedback.columns.values[1]: 'resultado'})
        feedback = feedback[['id_siniestro', 'resultado']]

    feedback['resultado'] = feedback['resultado'].str.upper()
    feedback = feedback.drop_duplicates(subset=['id_siniestro', 'resultado'])
    feedback = feedback[feedback['resultado'].isin(['POSITIVO', 'NEGATIVO'])]
    feedback = pd.merge(open_cases, feedback, how='inner', on='id_siniestro')
    df_cols = df.columns.values.tolist()
    df = df.append(feedback.drop('resultado', axis=1)).reset_index(drop=True)
    df = df[df_cols]
    feedback_1_list = feedback[feedback['resultado'] == 'POSITIVO']
    feedback_0_list = feedback[feedback['resultado'] == 'NEGATIVO']
    feedback_1_list = feedback_1_list[['id_siniestro']]
    feedback_0_list = feedback_0_list[['id_siniestro']]

delete_var = ['id_fiscal', 'id_poliza', 'cliente_codfiliacion']
for i in delete_var:
    if i in df:
        del df[i]

delete_var = [col for col in df if col.startswith('Unnamed')]
for i in delete_var:
    del df[i]

# FECHAS y CP CRUZADAS
if fecha_var:
    fecha_var += ['id_siniestro']
    fecha_file = df[fecha_var]
    for i in fecha_file.columns.values.tolist():
        if i != 'id_siniestro':
            fecha_file[i] = pd.to_datetime(fecha_file[i], format='%Y-%m-%d', errors='coerce')

    fecha_file.to_csv(STRING.training_auxiliar_fecha, sep=';', encoding='latin1',
                      index=False)
    fecha_var.remove('id_siniestro')

    fecha_file['fecha_diferencia_ref_polref'] = pd.Series(fecha_file['fecha_siniestro_ocurrencia'] -
                                                          fecha_file['hist_siniestro_otro_ultimo_fecha_ocurrencia'],
                                                          index=fecha_file.index).dt.days

    fecha_file.loc[fecha_file['fecha_diferencia_ref_polref'] < 0, 'fecha_diferencia_ref_polref'] = np.NaN
    fecha_file['fecha_diferencia_ref_polotra'] = pd.Series(fecha_file['fecha_siniestro_ocurrencia'] -
                                                           fecha_file
                                                           ['hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia'],
                                                           index=fecha_file.index).dt.days

    fecha_file.loc[fecha_file['fecha_diferencia_ref_polotra'] < 0, 'fecha_diferencia_ref_polotra'] = np.NaN
    fecha_file['fecha_diferencia_polref_polotra'] = pd.Series(
        fecha_file['hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia'] - fecha_file[
            'hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia'],
        index=fecha_file.index).dt.days

    fecha_file['fecha_diferencia_polref_polotra'] = fecha_file['fecha_diferencia_polref_polotra'].abs()
    fecha_file['fecha_dif_emision_sin_ref_polotra'] = pd.Series(fecha_file['fecha_poliza_efecto_natural'] -
                                                                fecha_file[
                                                                'hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia'],
                                                                index=fecha_file.index).dt.days

    fecha_file['fecha_dif_emision_sin_ref_polotra'] = fecha_file['fecha_dif_emision_sin_ref_polotra'].abs()

    fecha_file['fecha_dif_emision_sin_ref_polotra'] = pd.Series(fecha_file['fecha_poliza_efecto_natural'] -
                                                                fecha_file[
                                                                    'hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia'],
                                                                index=fecha_file.index).dt.days

    fecha_file['fecha_dif_emision_sin_ref_polotra'] = fecha_file['fecha_dif_emision_sin_ref_polotra'].abs()
    fecha_file = fecha_file.fillna(-1)

    for i in fecha_var:
            if i != 'id_siniestro':
                del fecha_file[i]
                del df[i]

    df = pd.merge(df, fecha_file, how='left', on='id_siniestro')
    del fecha_file

if cp:
    cp += ['id_siniestro']
    cp_file = df[cp]
    for i in cp_file.columns.values.tolist():
        cp_file[i] = pd.to_numeric(cp_file[i], errors='coerce')
        cp_file[i] = cp_file[i].fillna(0)
        cp_file[i] = cp_file[i].map(int)
    cp_file = cp_file.drop_duplicates(subset=['cliente_cp', 'hogar_cp', 'id_siniestro'])
    cp_file.to_csv(STRING.training_auxiliar_cp, sep=';', encoding='latin1', index=False)
    cp_file['cp_hogar_cliente_coincide'] = pd.Series(0, index=cp_file.index)
    cp_file.loc[cp_file['hogar_cp'] == cp_file['cliente_cp'], 'cp_cliente_coincide'] = 1
    del cp_file['hogar_cp']
    del cp_file['cliente_cp']
    del df['hogar_cp']
    del df['cliente_cp']
    print(cp_file)
    print(df)
    df = pd.merge(df, cp_file, how='left', on='id_siniestro')
    del cp_file

df = process_utils.process_utils.fillna_multioutput(df, not_consider=['id_siniestro'], n_estimators=100)
df = df.drop_duplicates()

# Define the variables that must to be deleted to because they are endogenous
delete_variables = ['hist_siniestro_actual_bbdd', 'hist_siniestro_actual_unidad_investigacion',
                    'hist_siniestro_actual_incidencia_tecnica', 'hist_siniestro_actual_incidencia_tecnica_positiva',
                    'hist_siniestro_actual_incidencias',
                    'pago_iban_blacklist', 'audit_poliza_entidad_legal'] + cp + fecha_var

# Delete Loss-Adjuster
for i in df.columns.values.tolist():
    if 'perit' in i:
        del df[i]

reduce_variables = ['cliente_iban_blacklist', 'cliente_id_fiscal_blacklist', 'pago_iban_blacklist']
if del_reduce_var:
    delete_variables = delete_variables + reduce_variables
else:
    for i in reduce_variables:
        df[i] = df[i] - 1
        df.loc[df[i] < 0, i] = 0

# We delete the variables that are endogenous
for i in delete_variables:
    if i == 'id_siniestro':
        delete_variables.remove(i)
    else:
        if i in df:
            del df[i]


# We need to check if a new Variable for where we do not have information before exists
# (It is impossible to train so we delete it)
len_col = len(df.index)
delete_cols = []
for i in df.columns.values.tolist():
    nan_values = df[i].isnull().sum()
    if nan_values == len_col:
        delete_cols.append(i)

for i in delete_cols:
    del df[i]


# First, we fill the categorical variables because they are not NaN
for i in STRING.fillna_vars:
    if i in df.columns.values.tolist():
        df[i] = df[i].fillna(0)

# Third, we fill the remaining
df = df.fillna(-1)

# ROBUST SCALE
# then compute the robust scale
df_base, params_rs = process_utils.process_utils.robust_scale(df, quantile_range=(10.0, 90.0))

# 7) PCA REDUCTION
df = df.dropna(subset=['id_siniestro'])
df = df.drop_duplicates()
columns_before_pca = df.columns.values.tolist()
df = df.reset_index(drop=True)

df, pca_components, params_scale = process_utils.process_utils.pca_reduction(df, variance=95.00)

# 8) APPEND BLACKLIST
df = process_utils.process_utils.append_blacklist(df)

if with_feedback:
    feedback_1_list['LIST1'] = pd.Series(1, index=feedback_1_list.index)
    df = pd.merge(df, feedback_1_list, how='left', on='id_siniestro')
    df.loc[df['LIST1'] == 1, 'FRAUDE'] = 1
    del df['LIST1']

    feedback_0_list['LIST0'] = pd.Series(1, index=feedback_0_list.index)
    df = pd.merge(df, feedback_0_list, how='left', on='id_siniestro')
    df = df[df['LIST0'].isnull()].reset_index(drop=True)

    feedback_0 = df[df['LIST0'] == 1].reset_index(drop=True)
    feedback_0['FRAUDE'] = pd.Series(0, index=feedback_0.index)
    feedback_0 = feedback_0.drop_duplicates(subset='id_siniestro')
    del feedback_0['LIST0']
    del df['LIST0']

    feedback_1_list = feedback_1_list[['id_siniestro']].values.tolist()
    feedback_0_list = feedback_0_list[['id_siniestro']].values.tolist()


normal = df[df['FRAUDE'] == 0].drop_duplicates(subset='id_siniestro').reset_index(drop=True)
anomaly = df[df['FRAUDE'] == 1].drop_duplicates(subset='id_siniestro').reset_index(drop=True)

if reduce_sample is not None:

    # We load Date of claims and filter by reduce_sample
    fecha_ocurrencia = pd.read_csv(STRING.training_auxiliar_fecha, sep=';',
                                   encoding='latin1')
    fecha_ocurrencia = fecha_ocurrencia[['id_siniestro', 'fecha_siniestro_ocurrencia']]
    fecha_ocurrencia['fecha_siniestro_ocurrencia'] = pd.to_datetime(fecha_ocurrencia['fecha_siniestro_ocurrencia'],
                                                                    format='%Y-%m-%d', errors='coerce')
    fecha_ocurrencia = fecha_ocurrencia.dropna(subset=['fecha_siniestro_ocurrencia'])
    fecha_ocurrencia.loc[fecha_ocurrencia[
                             'fecha_siniestro_ocurrencia'] < date_threshold, 'fecha_siniestro_ocurrencia'] = np.NaN
    fecha_ocurrencia = fecha_ocurrencia.dropna(subset=['fecha_siniestro_ocurrencia'])
    del fecha_ocurrencia['fecha_siniestro_ocurrencia']

    # We reduce the normal cases
    normal = pd.merge(fecha_ocurrencia, normal, how='left', on='id_siniestro')
    normal = normal.dropna(subset=['FRAUDE'])
    del fecha_ocurrencia


"----------------------------------------UNSUPERVISED MODEL-------------------------------------------------------"
# 1) First we calculate the Fraud Score for each model
# oversample_times = round(len(normal.index) / len(anomaly.index))
normal = normal.sort_values(by=['id_siniestro'], ascending=True).reset_index(drop=True)
anomaly = anomaly.sort_values(by=['id_siniestro'], ascending=True).reset_index(drop=True)

kmeans_fscore = uns.mini_batch_kmeans_tunning(normal, anomaly, oversample_times, max_iter=par.get("max_iter"),
                                              n_clusters=par.get("n_clusters"),
                                              batch_size_range=par.get("batch_size_range"))

# 2) We take the params that are related to the max Fraud Score
kmeans_fscore = kmeans_fscore[kmeans_fscore['fraud_score'] == kmeans_fscore['fraud_score'].max()].reset_index(
    drop=True)

max_iter = kmeans_fscore.at[0, 'max_iter']
batch_size = kmeans_fscore.at[0, 'batch_size']
n_clusters = kmeans_fscore.at[0, 'n_clusters']

print('Mean Shift with ' + 'iters ' + str(max_iter) + 'batch ' + str(batch_size) + 'cluster ' + str(n_clusters))
if oversample_times is None:
    normal, anomaly = model_utils.oversample_unsupervised(normal, anomaly)
else:
    anomaly = anomaly.append([anomaly] * oversample_times, ignore_index=True)

_, _, _, comparative = mini_batch_kmeans.mini_batch_kmeans(normal, anomaly, n_clusters=n_clusters,
                                                           max_iter=max_iter, batch_size=batch_size)

# We get the Clusters of the best Model
comparative = comparative.drop_duplicates(subset='id_siniestro')
comparative = comparative[comparative['id_siniestro'] >= 0]

# We send the results to a Control Point of the Unsupervised Model


os.makedirs(os.path.dirname(STRING.monitoring_path_no_supervisado), exist_ok=True)
comparative.to_csv(STRING.monitoring_no_supervisado_uns_class, sep=';', encoding='latin1', index=False)
comparative = comparative.drop_duplicates(subset=['id_siniestro', 'FRAUDE', 'Clusters', 'FRAUDE_Clusters'])
comparative['id_siniestro'] = comparative['id_siniestro'].map(int)

"---------------------------------------RELABEL BY FEEDBACK AND SAMPLE REDUCE--------------------------------------"
if reduce_sample is not None:

    # We load Date of claims and filter by reduce_sample
    fecha_ocurrencia = pd.read_csv(STRING.training_auxiliar_fecha, sep=';',
                                   encoding='latin1')
    fecha_ocurrencia = fecha_ocurrencia[['id_siniestro', 'fecha_siniestro_ocurrencia']]
    fecha_ocurrencia['fecha_siniestro_ocurrencia'] = pd.to_datetime(fecha_ocurrencia['fecha_siniestro_ocurrencia'],
                                                                    format='%Y-%m-%d', errors='coerce')

    fecha_ocurrencia.loc[fecha_ocurrencia[
                             'fecha_siniestro_ocurrencia'] < date_threshold, 'fecha_siniestro_ocurrencia'] = np.NaN
    fecha_ocurrencia = fecha_ocurrencia.dropna(subset=['fecha_siniestro_ocurrencia'])
    del fecha_ocurrencia['fecha_siniestro_ocurrencia']

    # We reduce only the normal that are FRAUDE_Clusters = 0
    comparative_reduced = comparative[['id_siniestro', 'FRAUDE_Clusters']]
    normal = pd.merge(normal, comparative_reduced, how='left', on='id_siniestro')
    del comparative_reduced

    normal_cluster_0 = pd.merge(fecha_ocurrencia, normal, how='left', on='id_siniestro')
    normal_cluster_0 = normal_cluster_0.dropna(subset=['FRAUDE'], axis=0)
    del normal_cluster_0['FRAUDE_Clusters']
    del fecha_ocurrencia

    # We merge again normal_cluster_0 and the FRAUD = 1
    normal = normal_cluster_0.copy()

if with_feedback:
    # We add Feedback_0 to the Comparative table
    feedback_0['id_siniestro'] = feedback_0['id_siniestro'].map(int)

    feedback_0_comparative = feedback_0[['id_siniestro']]
    feedback_0_comparative['FRAUDE'] = pd.Series(0, index=feedback_0_comparative.index)
    feedback_0_comparative['Clusters'] = pd.Series(-1, index=feedback_0_comparative.index)
    feedback_0_comparative['FRAUDE_Clusters'] = pd.Series(0, index=feedback_0_comparative.index)
    comparative = pd.concat([comparative, feedback_0_comparative], axis=0)
    del feedback_0_comparative
    normal = pd.concat([normal, feedback_0], axis=0)

"---------------------------------------TRAINING MODEL----------------------------------------------------------------"
normal = normal.drop_duplicates(subset='id_siniestro')
normal = normal[normal['id_siniestro'] > 0]
anomaly = anomaly.drop_duplicates(subset='id_siniestro')
anomaly = anomaly[anomaly['id_siniestro'] > 0]
normal = normal.dropna(how='all', axis=1)
normal = normal.dropna(how='all', axis=1)
normal['id_siniestro'] = normal['id_siniestro'].map(int)
anomaly['id_siniestro'] = anomaly['id_siniestro'].map(int)

# Train - Test - Valid
train_t, test_t, valid_t = train_test_utils.train_test.training_test_valid(normal, anomaly)
train_t['id_siniestro'] = train_t['id_siniestro'].map(int)
test_t['id_siniestro'] = test_t['id_siniestro'].map(int)
valid_t['id_siniestro'] = valid_t['id_siniestro'].map(int)
comparative['id_siniestro'] = comparative['id_siniestro'].map(int)
train_t = pd.merge(train_t, comparative.drop(['FRAUDE'], axis=1), how='left', on='id_siniestro')
test_t = pd.merge(test_t, comparative.drop(['FRAUDE'], axis=1), how='left', on='id_siniestro')
valid_t = pd.merge(valid_t, comparative.drop(['FRAUDE'], axis=1), how='left', on='id_siniestro')
train_t = train_t.drop_duplicates(subset='id_siniestro')
test_t = test_t.drop_duplicates(subset='id_siniestro')
valid_t = valid_t.drop_duplicates(subset='id_siniestro')

comparative = comparative[['id_siniestro', 'FRAUDE', 'FRAUDE_Clusters']]

# LABELS
labels = ['FRAUDE', 'Clusters', 'FRAUDE_Clusters']
selected_label = 'FRAUDE_Clusters'
labels.remove(selected_label)
for i in labels:
    del train_t[i]
    del test_t[i]
    del valid_t[i]

train_t = train_t.reset_index(drop=True)
test_t = test_t.reset_index(drop=True)
valid_t = valid_t.reset_index(drop=True)

valid_t[selected_label] = valid_t[selected_label].map(int)
test_t[selected_label] = test_t[selected_label].map(int)

# XTRAIN-YTRAIN
yTrain = train_t[[selected_label]]
yTrain[selected_label] = yTrain[selected_label].map(int)

xTrain_base = train_t.drop(selected_label, axis=1).copy()
yTrain_base = yTrain.copy()
xTrain_control = train_t.drop(selected_label, axis=1).copy()
yTrain_control = yTrain.copy()

Valid = pd.concat([train_t, valid_t], axis=0)
Valid = pd.merge(Valid, comparative[['id_siniestro']], on='id_siniestro', how='left')
Valid = Valid.dropna()

# SELECTED BEST MODEL: BASE MODEL PERFORMANCE
if base_sampling is None:
    class_weight = 'balanced_subsample'
    xTrain_base = xTrain_base.drop(['id_siniestro'], axis=1)
elif base_sampling == 'ALLKNN':
    xTrain_base, yTrain_base = model_utils.under_sampling(xTrain_base.drop(['id_siniestro'], axis=1), yTrain_base)
    class_weight = None
else:
    xTrain_base, yTrain_base = model_utils.over_sampling(xTrain_base.drop(['id_siniestro'], axis=1), yTrain_base,
                                                         model=base_sampling)
    class_weight = None

min_sample_leaf = round((len(xTrain_base.index)) * 0.01)
min_sample_split = min_sample_leaf * 10
max_features = 'sqrt'

fileModel = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=bootstrap,
                                          min_samples_leaf=min_sample_leaf,
                                          min_samples_split=min_sample_split,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth, max_features=max_features,
                                          oob_score=oob_score,
                                          random_state=531, verbose=1, class_weight=class_weight,
                                          n_jobs=1)

fileModel = fileModel.fit(xTrain_base.values, yTrain_base.values)

if base_sampling is None:
    cv = model_selection.StratifiedKFold(n_splits=5, random_state=None)
else:
    cv = 5

y_pred_score = model_selection.cross_val_predict(fileModel,
                                                 Valid.drop([selected_label] + ['id_siniestro'], axis=1).values,
                                                 Valid[[selected_label]].values, cv=cv, method='predict_proba')

y_pred_score = np.delete(y_pred_score, 0, axis=1)
y_hat_test = (y_pred_score > threshold_models).astype(int)
y_hat_test = y_hat_test.tolist()
y_hat_test = [item for sublist in y_hat_test for item in sublist]
recall_base = metrics.recall_score(y_pred=y_hat_test, y_true=Valid[selected_label].values)
precision_base = metrics.precision_score(y_pred=y_hat_test, y_true=Valid[selected_label].values)
fbeta_value_base = metrics.fbeta_score(y_pred=y_hat_test, y_true=Valid[selected_label].values, beta=beta)

# MODELO BASE
fileModel = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=bootstrap,
                                          min_samples_leaf=min_sample_leaf,
                                          min_samples_split=min_sample_split,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth, max_features=max_features,
                                          oob_score=oob_score,
                                          random_state=531, verbose=1, class_weight=class_weight,
                                          n_jobs=1)
fileModelBase = fileModel.fit(xTrain_base.values, yTrain_base.values)

# SELECTED BEST MODEL: CONTROL MODEL PERFORMANCE
if control_sampling is None:
    class_weight = 'balanced_subsample'
    xTrain_control = xTrain_control.drop(['id_siniestro'], axis=1)

elif control_sampling == 'ALLKNN':
    xTrain_control, yTrain_control = model_utils.under_sampling(xTrain_control.drop(['id_siniestro'], axis=1),
                                                                yTrain_control)
    class_weight = None
else:
    xTrain_control, yTrain_control = model_utils.over_sampling(xTrain_control.drop(['id_siniestro'], axis=1),
                                                               yTrain_control, model=control_sampling)
    class_weight = None

min_sample_leaf = round((len(xTrain_control.index)) * 0.01)
min_sample_split = min_sample_leaf * 10
max_features = 'sqrt'

fileModel = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=bootstrap,
                                          min_samples_leaf=min_sample_leaf,
                                          min_samples_split=min_sample_split,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth, max_features=max_features,
                                          oob_score=oob_score,
                                          random_state=531, verbose=1, class_weight=class_weight,
                                          n_jobs=1)

fileModel = fileModel.fit(xTrain_control.values, yTrain_control.values)
if base_sampling is None:
    cv = model_selection.StratifiedKFold(n_splits=5, random_state=None)
else:
    cv = 5
y_pred_score = model_selection.cross_val_predict(fileModel,
                                                 Valid.drop([selected_label] + ['id_siniestro'], axis=1).values,
                                                 Valid[[selected_label]].values, cv=cv, method='predict_proba')

y_pred_score = np.delete(y_pred_score, 0, axis=1)
y_hat_test = (y_pred_score > threshold_models).astype(int)
y_hat_test = y_hat_test.tolist()
y_hat_test = [item for sublist in y_hat_test for item in sublist]
recall_control = metrics.recall_score(y_pred=y_hat_test, y_true=Valid[selected_label].values)
precision_control = metrics.precision_score(y_pred=y_hat_test, y_true=Valid[selected_label].values)
fbeta_value_control = metrics.fbeta_score(y_pred=y_hat_test, y_true=Valid[selected_label].values, beta=beta)

# MODELO CONTROL
fileModel = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=bootstrap,
                                          min_samples_leaf=min_sample_leaf,
                                          min_samples_split=min_sample_split,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth, max_features=max_features,
                                          oob_score=oob_score,
                                          random_state=531, verbose=1, class_weight=class_weight,
                                          n_jobs=1)
fileModelControl = fileModel.fit(xTrain_control.values, yTrain_control.values)

# GUARDAR RESULTADOS
column_performance = ['THRESHOLD_BASE', 'PRECISION_BASE',
                      'RECALL_BASE', 'FBETA_BASE', 'THRESHOLD_CONTROL', 'PRECISION_CONTROL', 'RECALL_CONTROL',
                      'FBETA_CONTROL']

performance_list = [[threshold_models, precision_base, recall_base, fbeta_value_base, threshold_models,
                     precision_control, recall_control, fbeta_value_control]]

performance_file = pd.DataFrame(performance_list, columns=column_performance)

os.makedirs(os.path.dirname(STRING.monitoring_path_supervisado), exist_ok=True)
performance_file.to_csv(STRING.monitoring_supervisado_performance, index=False, encoding='latin1',
                        sep=';')

# GUARDAR Pickle
dict_models = {'modelo_base': fileModelBase, 'modelo_control': fileModelControl, 'precision_base': precision_base,
               'precision_control': precision_control, 'params_rs': params_rs,
               'pca_components': pca_components, 'feedback_1': feedback_1_list,
               'feedback_0': feedback_0_list,
               'columns_pca': columns_before_pca, 'params_scale': params_scale}

joblib.dump(dict_models,
            os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                         "fraud_home", "models", "model_zfinder", "model_zfinder.pkl"))
