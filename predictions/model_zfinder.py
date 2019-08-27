from sklearn.externals import joblib
import os
import numpy as np
import pandas as pd

from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home.red_flags import red_flags
from fraud_home.configs.config import parameters as par
from fraud_home.resources.fraud_home.checklist import ChecklistsObligatorias
from fraud_home.resources.fraud_home.process_utils import process_utils


def pre_process(data_input):

    # Paramters
    with_feedback = par.get('with_feedback')
    fecha_var = par.get("fecha_var")
    cp = par.get("cp")
    del_reduce_var = par.get("del_reduce_var")

    dict_models = joblib.load(
        os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     "fraud_home", "models", "model_zfinder", "model_zfinder.pkl"))

    param_rs = dict_models.get("params_rs")
    pca_components = dict_models.get("pca_components")
    feedback_1_list = dict_models.get("feedback_1")
    feedback_0_list = dict_models.get("feedback_0")
    columns_before_pca = dict_models.get("columns_pca")
    params_scale = dict_models.get("params_scale")

    del dict_models

    if with_feedback:
        feedback_0_list = [item for sublist in feedback_0_list for item in sublist]
        feedback_1_list = [item for sublist in feedback_1_list for item in sublist]
        feedback_0_list = pd.DataFrame(feedback_0_list, columns=['id_siniestro'])
        feedback_1_list = pd.DataFrame(feedback_1_list, columns=['id_siniestro'])

        feedback = pd.concat([feedback_0_list, feedback_1_list], axis=0).reset_index(drop=True)
        feedback['DELETE'] = pd.Series(1, index=feedback.index)
        input_data = pd.merge(data_input, feedback, how='left', on='id_siniestro')
        input_data = input_data[input_data['DELETE'] != 1].reset_index(drop=True)
        del input_data['DELETE']
        del feedback

    delete_var = ['id_fiscal', 'id_poliza', 'cliente_codfiliacion']
    for i in delete_var:
        if i in input_data:
            del input_data[i]

    delete_var = [col for col in input_data if col.startswith('Unnamed')]
    for i in delete_var:
        del input_data[i]

    # FECHAS y CP CRUZADAS
    if fecha_var:
        fecha_var += ['id_siniestro']
        fecha_file = input_data[fecha_var]
        for i in fecha_file.columns.values.tolist():
            if i != 'id_siniestro':
                fecha_file[i] = pd.to_datetime(fecha_file[i], format='%Y-%m-%d', errors='coerce')

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
                del input_data[i]

        input_data = pd.merge(input_data, fecha_file, how='left', on='id_siniestro')
        del fecha_file

    if cp:
        cp += ['id_siniestro']
        cp_file = input_data[cp]
        for i in cp_file.columns.values.tolist():
            cp_file[i] = pd.to_numeric(cp_file[i], errors='coerce')
            cp_file[i] = cp_file[i].fillna(0)
            cp_file[i] = cp_file[i].map(int)
        cp_file['cp_hogar_cliente_coincide'] = pd.Series(0, index=cp_file.index)
        cp_file.loc[cp_file['hogar_cp'] == cp_file['cliente_cp'], 'cp_cliente_coincide'] = 1
        del cp_file['hogar_cp']
        del cp_file['cliente_cp']
        del input_data['hogar_cp']
        del input_data['cliente_cp']
        input_data = pd.merge(input_data, cp_file, how='left', on='id_siniestro')
        del cp_file

    input_data = process_utils.fillna_multioutput(input_data, not_consider=['id_siniestro'])

    input_data = input_data.drop_duplicates()

    # Define the variables that must to be deleted to because they are endogenous
    delete_variables = ['hist_siniestro_actual_bbdd', 'hist_siniestro_actual_unidad_investigacion',
                        'hist_siniestro_actual_incidencia_tecnica', 'hist_siniestro_actual_incidencia_tecnica_positiva',
                        'hist_siniestro_actual_incidencias',
                        'pago_iban_blacklist', 'audit_poliza_entidad_legal'] + cp + fecha_var

    reduce_variables = ['cliente_iban_blacklist', 'cliente_id_fiscal_blacklist', 'pago_iban_blacklist']
    if del_reduce_var:
        delete_variables = delete_variables + reduce_variables
    else:
        for i in reduce_variables:
            input_data[i] = input_data[i] - 1
            input_data.loc[input_data[i] < 0, i] = 0

    # We delete the variables that are endogenous
    for i in delete_variables:
        if i == 'id_siniestro':
            delete_variables.remove(i)
        else:
            if i in input_data:
                del input_data[i]

    # CHECK IF DIARIO & MENSUAL HAS THE SAME COLUMNS (IF NOT WE ADD)
    input_col = input_data.columns.values.tolist()
    add_cols = []
    for col in columns_before_pca:
        if col not in input_col:
            add_cols.append(col)

    for col in add_cols:
        input_data[col] = pd.Series(0, index=input_data.index)

    # First, we fill the categorical variables because they are not NaN
    for i in STRING.fillna_vars:
        if i in input_data.columns.values.tolist():
            input_data[i] = input_data[i].fillna(0)

    input_data = input_data[columns_before_pca]
    # Third, we fill the remaining
    input_data = input_data.fillna(-1)

    # ROBUST SCALE
    param_rs = pd.DataFrame(param_rs, columns=['column_name', 'center', 'scale'])
    params_scale = pd.DataFrame(params_scale, columns=['column_name', 'mean', 'var'])
    columns_to_scale = params_scale['column_name'].tolist()
    for i in columns_to_scale:
        center = param_rs.loc[param_rs['column_name'] == i, 'center'].iloc[0]
        scale = param_rs.loc[param_rs['column_name'] == i, 'scale'].iloc[0]
        mean = params_scale.loc[params_scale['column_name'] == i, 'mean'].iloc[0]
        var = params_scale.loc[params_scale['column_name'] == i, 'var'].iloc[0]
        if scale != 0 and var != 0:
            input_data[i] = (input_data[i] - center) / scale
            input_data[i] = (input_data[i] - mean) / np.sqrt(var)
        else:
            input_data[i] = pd.Series(-1, index=input_data.index)
        input_data[i] = input_data[i].map(float)

    # PCA
    input_data = input_data.reset_index(drop=True)
    siniestro_df = input_data[['id_siniestro']]
    input_data = pca_components.transform(input_data.drop('id_siniestro', axis=1))
    input_data = pd.DataFrame(input_data)
    output = pd.concat([input_data, siniestro_df], axis=1)
    return output


def prediction(output):
    dict_models = joblib.load(
        os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     "fraud_home", "models", "model_zfinder", "model_zfinder.pkl"))

    precision_base = dict_models.get("precision_base")
    precision_control = dict_models.get("precision_control")
    threshold_models = par.get("threshold_models")

    if precision_base >= precision_control:
        model_base = dict_models.get("modelo_base")
        model_control = dict_models.get("modelo_control")
    else:
        model_base = dict_models.get("model_control")
        model_control = dict_models.get("model_base")
    del dict_models

    prediction_base = model_base.predict_proba(output.drop('id_siniestro', axis=1).values)
    prediction_base = np.delete(prediction_base, 0, axis=1)
    prediction_base = pd.DataFrame(prediction_base, columns=['probabilidad_base'], index=output.index)
    prediction_base['Treshold_base'] = np.where(prediction_base['probabilidad_base'] > threshold_models, 1, 0)
    prediction_base = pd.concat([output['id_siniestro'], prediction_base], axis=1)

    prediction_control = model_control.predict_proba(output.drop('id_siniestro', axis=1).values)
    prediction_control = np.delete(prediction_control, 0, axis=1)
    prediction_control = pd.DataFrame(prediction_control, columns=['probabilidad_control'], index=output.index)
    prediction_control['Treshold_control'] = np.where(prediction_control['probabilidad_control'] > threshold_models, 1,
                                                      0)

    output = pd.concat([prediction_base, prediction_control], axis=1)

    return output


def post_process(output):
    n_random = par.get("n_random")
    refactor_prob = par.get("refactor_probability")
    entity = par.get("entity")

    # LOAD OPEN CASES
    open_cases = pd.read_csv(
        STRING.etl_diaria,
        sep=';', encoding='latin1', dtype={'id_siniestro': int})

    output = output.rename(columns={'probabilidad_base': 'probabilidad'})
    del output['probabilidad_control']

    # CONCAT PROBABILITY
    df_proba_conc = pd.merge(output,
                             open_cases[['id_siniestro', 'po_res_indem', 'po_pago_indemnizacion_importe_neto_count',
                                         'id_poliza', 'id_fiscal', 'fecha_siniestro_ocurrencia']])
    df_proba_conc = df_proba_conc.sort_values(by=['id_siniestro', 'po_res_indem'], ascending=[True, False])
    df_proba_conc = df_proba_conc.drop_duplicates(subset='id_siniestro', keep='first')
    df_proba_conc['proba_ponderada'] = (df_proba_conc['po_res_indem']) * df_proba_conc['probabilidad']
    df_proba_conc = df_proba_conc.sort_values(by=['probabilidad', 'proba_ponderada'], ascending=[False, False])
    df_proba_conc['id_siniestro'] = df_proba_conc['id_siniestro'].map(int)
    df_proba_conc['probabilidad'] = df_proba_conc['probabilidad'].map(float)

    # RED FLAGS
    rf = red_flags(open_cases)
    rf.to_csv(STRING.red_flag_path, sep=';', encoding='latin1')
    df_proba_conc = pd.merge(df_proba_conc, rf, how='left', on='id_siniestro')

    # LOAD PROBABILITY FIL with +/-0.1 probability range
    try:
        base_prob = pd.read_csv(STRING.base_probabilidad, sep=';', dtype={'id_siniestro': int, 'probabilidad': float},
                                encoding='latin1')
    except FileNotFoundError:
        base_prob = pd.DataFrame(columns=['id_siniestro', 'probabilidad_anterior']
                                 )

    df_proba_conc = pd.merge(df_proba_conc, base_prob, how='left', on=['id_siniestro'])
    df_proba_conc['probabilidad_anterior'] = df_proba_conc['probabilidad_anterior'].fillna(0)

    df_proba_conc['indicador'] = pd.Series(0, index=df_proba_conc.index)
    df_proba_conc.loc[(df_proba_conc['probabilidad'] > df_proba_conc['probabilidad_anterior'] + 0.1) |
                      (df_proba_conc['probabilidad'] < df_proba_conc['probabilidad_anterior'] - 0.1),
                      'indicador'] = 1

    df_proba_conc = df_proba_conc[df_proba_conc['indicador'] == 1]
    del df_proba_conc['indicador']

    df_proba_conc.loc[df_proba_conc['probabilidad_anterior'] == 0, 'probabilidad_anterior'] = np.NaN

    # Now we will keep only the cases where Treshold = Threshold_control. But first we will save these cases for us
    df_proba_conc.to_csv(STRING.probabilidad_ambos_modelos, sep=';', index=False,
                         encoding='latin1')

    # Randomly selection
    df_random = df_proba_conc.copy()
    df_random = df_random.sample(n=n_random)
    df_random.to_csv(STRING.control_evaluation, sep=';', index=False)

    df_random = df_random[df_random['probabilidad'] < 0.50]
    df_random['probabilidad'] = np.random.uniform(0.50, 0.55, df_random.shape[0])

    # CHECKLIST
    checklist_file = open_cases[['id_siniestro', 'checklist6a', 'checklist5_poliza', 'checklist5_nif',
                                 'checklist6a_PP', 'checklist_6b', 'checklist_7',
                                 'checklist_14_coberturas_repetidas', 'checklist_14_siniestros_involucrados']]

    checklist_2 = ChecklistsObligatorias.checklist2()
    checklist_3 = ChecklistsObligatorias.checklist3()
    checklist_4 = ChecklistsObligatorias.checklist4()

    checklist_file = pd.merge(checklist_file, checklist_2, how='left', on='id_siniestro')
    checklist_file = pd.merge(checklist_file, checklist_3, how='left', on='id_siniestro')
    checklist_file = pd.merge(checklist_file, checklist_4, how='left', on='id_siniestro')

    checklist_file = checklist_file.fillna(0)
    checklist_file['CL_indicator'] = pd.Series(0, index=checklist_file.index)
    checklist_var_obligatorias = ['checklist2', 'po_res_indem_mayor_5000', 'checklist6a',
                                  'checklist5_poliza', 'checklist5_nif', 'checklist_7']
    for col in checklist_var_obligatorias:
        checklist_file.loc[checklist_file[col] >= 1, 'CL_indicator'] = 1
    checklist_mayor1 = ['checklist4_poliza', 'checklist4_nif', 'checklist_6b']
    for col in checklist_mayor1:
        checklist_file.loc[checklist_file[col] > 1, 'CL_indicator'] = 1
    checklist_valoracion = ['checklist_14_siniestros_involucrados']
    for col in checklist_valoracion:
        checklist_file.loc[checklist_file[col] >= 4, 'CL_indicator'] = 1

    checklist_file.to_csv(STRING.checklist,
                          index=False, sep=';', encoding='latin1')

    df_proba_conc['id_siniestro'] = df_proba_conc['id_siniestro'].map(int)
    df_proba_conc = pd.merge(df_proba_conc, checklist_file, on='id_siniestro', how='left')
    for i in df_proba_conc.columns.values.tolist():
        if i.startswith('Unnam'):
            del df_proba_conc[i]

    # REGLA DE SELECCION
    df_proba_conc = df_proba_conc[((df_proba_conc['Treshold_base'] == 1) & (df_proba_conc['Treshold_control'] == 1)) |
                                  (df_proba_conc['RF_indicator'] == 1) | (df_proba_conc['CL_indicator'] == 1)]
    df_proba_conc = pd.concat([df_proba_conc, df_random], axis=0)
    df_proba_conc = df_proba_conc.sort_values(by=['probabilidad'], ascending=False)
    df_proba_conc = df_proba_conc.drop_duplicates(subset=['id_siniestro'], keep='first')
    del df_proba_conc['Treshold_base']
    del df_proba_conc['Treshold_control']

    # First we save the unnormalized probability
    df_proba_conc.to_csv(STRING.probabilidad_unnormalized, sep=';', index=False)

    # As we have a dense probability we reescalate using 0.8 as max
    df_proba_conc['probabilidad'] = df_proba_conc['probabilidad'] / refactor_prob
    df_proba_conc.loc[df_proba_conc['probabilidad'] > 1, 'probabilidad'] = 1
    df_proba_conc['probabilidad_anterior'] = df_proba_conc['probabilidad_anterior'] / refactor_prob
    df_proba_conc.loc[df_proba_conc['probabilidad_anterior'] > 1, 'probabilidad_anterior'] = 1

    # We send a copy for us
    df_proba_conc['FECHA_ALERTA'] = pd.Series(str(STRING.DAY), index=df_proba_conc.index)
    df_proba_conc.to_csv(STRING.probabilidad_normalized, sep=';',
                         index=False)

    del df_proba_conc['proba_ponderada']
    del checklist_file

    df_proba_conc = df_proba_conc.rename(columns={'po_pago_indemnizacion_importe_neto_count': 'po_pago_indem_neto'})

    # We order file columns
    output_final = df_proba_conc[['id_siniestro', 'id_poliza', 'id_fiscal', 'probabilidad',
                                  'probabilidad_anterior', 'po_res_indem',
                                  'po_pago_indem_neto', 'po_res_indem_mayor_5000', 'fecha_siniestro_ocurrencia',
                                  'FECHA_ALERTA', 'checklist2', 'checklist3a', 'checklist3b', 'checklist3c',
                                  'checklist4_poliza',
                                  'checklist4_nif',
                                  'checklist5_poliza', 'checklist5_nif', 'checklist6a', 'checklist6a_PP',
                                  'checklist_6b', 'checklist_7', 'checklist_14_coberturas_repetidas',
                                  'checklist_14_siniestros_involucrados',
                                  'RF_fecha_diferencia_siniestro_efecto',
                                  'RF_fecha_diferencia_siniestro_efecto_5',
                                  'RF_fecha_diferencia_siniestro_efecto_15',
                                  'RF_fecha_diferencia_siniestro_efecto_30',
                                  'RF_fecha_diferencia_siniestro_emision',
                                  'RF_fecha_diferencia_siniestro_emision_5',
                                  'RF_fecha_diferencia_siniestro_emision_15',
                                  'RF_fecha_diferencia_siniestro_emision_30',
                                  'RF_fecha_siniestro_ocurrencia',
                                  'RF_fecha_poliza_emision', 'RF_fecha_poliza_efecto_natural',
                                  'RF_fecha_ocurrencia_entre_efecto_emision',
                                  'RF_fecha_diferencia_siniestro_comunicacion',
                                  'RF_retraso_comunicacion',
                                  'RF_mediador',
                                  'RF_indicator', 'CL_indicator'
                                  ]]

    # We save in the base_prob base the new claims
    df_proba_base = df_proba_conc[['id_siniestro', 'probabilidad']]
    df_proba_base = df_proba_base.rename(columns={'probabilidad': 'probabilidad_anterior'})
    base_prob = pd.concat([base_prob, df_proba_base], axis=0, ignore_index=True)
    base_prob = base_prob.drop_duplicates(subset=['id_siniestro'], keep='last')
    base_prob.to_csv(STRING.base_probabilidad, sep=';', index=False)

    output_final.to_csv(STRING.probabilidad_output, index=False, sep=';', decimal=',')

    return None


if __name__ == '__main__':
    input_data = pd.read_csv(STRING.etl_diaria, sep=';', encoding='latin1')
    output1 = pre_process(input_data)
    output2 = prediction(output1)
    output_final = post_process(output2)
