import pandas as pd

from fraud_home.resources.fraud_home import STRING


def red_flags(test_file: pd.DataFrame):

    # Difference claims and policy effect/emission
    test_file = test_file[
        ['id_siniestro', 'fecha_diferencia_siniestro_efecto', 'fecha_diferencia_siniestro_efecto_5',
         'fecha_diferencia_siniestro_efecto_15', 'fecha_diferencia_siniestro_efecto_30',
         'fecha_diferencia_siniestro_emision', 'fecha_diferencia_siniestro_emision_5',
         'fecha_diferencia_siniestro_emision_15', 'fecha_diferencia_siniestro_emision_30',
         'fecha_siniestro_ocurrencia',
         'fecha_poliza_emision', 'fecha_poliza_efecto_natural',
         'fecha_diferencia_siniestro_comunicacion'
         ]]

    policy_file = pd.read_csv(STRING.poliza_input_prediction, sep=',',
                              encoding='utf-8', quotechar='"')
    policy_file = policy_file[['audit_siniestro_referencia', 'poliza_cod_intermediario']]
    policy_file = policy_file.rename(
        columns={'audit_siniestro_referencia': 'id_siniestro', 'poliza_cod_intermediario': 'id_mediador'})
    policy_file['id_siniestro'] = policy_file['id_siniestro'].map(int)
    test_file['id_siniestro'] = test_file['id_siniestro'].map(int)
    test_file = pd.merge(test_file, policy_file, how='left', on='id_siniestro')

    test_file = test_file.dropna(subset=['id_siniestro'])
    test_file['id_mediador'] = test_file['id_mediador'].fillna(-1)

    # Occurance between effect and emision
    for i in ['fecha_siniestro_ocurrencia', 'fecha_poliza_emision', 'fecha_poliza_efecto_natural']:
        test_file[i] = pd.to_datetime(test_file[i], format='%Y-%m-%d', errors='coerce')

    test_file['fecha_ocurrencia_entre_efecto_emision'] = pd.Series(0, index=test_file.index)
    test_file.loc[(test_file['fecha_poliza_emision'] <= test_file['fecha_siniestro_ocurrencia']) & (
            test_file['fecha_siniestro_ocurrencia'] <= test_file[
             'fecha_poliza_efecto_natural']), 'fecha_ocurrencia_entre_efecto_emision'] = 1

    test_file.loc[(test_file['fecha_poliza_efecto_natural'] <= test_file['fecha_siniestro_ocurrencia']) &
                  (test_file['fecha_siniestro_ocurrencia'] <= test_file['fecha_poliza_emision']),
                  'fecha_ocurrencia_entre_efecto_emision'] = 1

    # Comunication and occurance difference
    test_file['retraso_comunicacion'] = pd.Series(0, index=test_file.index)
    test_file.loc[test_file['fecha_diferencia_siniestro_comunicacion'] >= 15, 'retraso_comunicacion'] = 1

    # If mediador
    test_file['mediador'] = pd.Series(0, index=test_file.index)
    test_file['id_mediador'] = test_file['id_mediador'].map(int)
    test_file.loc[test_file['id_mediador'] == 62659, 'mediador'] = 1

    # Indicator of RF
    test_file['indicator'] = pd.Series(0, index=test_file.index)

    # test_file.loc[test_file['fecha_diferencia_siniestro_efecto'] <= 30, 'indicator'] = 1
    test_file.loc[test_file['fecha_diferencia_siniestro_emision'] <= 30, 'indicator'] = 1
    test_file.loc[test_file['retraso_comunicacion'] == 1, 'indicator'] = 1
    test_file.loc[test_file['fecha_ocurrencia_entre_efecto_emision'] == 1, 'indicator'] = 1
    test_file.loc[test_file['mediador'] == 1, 'indicator'] = 1

    test_file = test_file.add_prefix('RF_')
    test_file = test_file.rename(columns={'RF_id_siniestro': 'id_siniestro'})

    return test_file
