import os

import pandas as pd

from fraud_home.resources.fraud_home import STRING


class ChecklistsObligatorias:
    """
    Si al menos uno se activa
    """

    def checklist1(self):
        """Conocido Clan / familia reincidente"""
        # Este no se puede construir. Se deriva de la experiencia del tramitador

    @staticmethod
    def checklist2():
        """
        Implicado con antecedentes de fraude recurrente y en conocimiento del Tramitador
        :return: This return a Dataframe with the columns 'id_siniestro', 'checklist2', where 'checklist2' counts the 
        number of times an ID_FISCAL appears in the Blacklist
        """

        df_test_id = pd.read_csv(STRING.id_input_prediction, sep=',',
                                 encoding='utf-8', quotechar='"')
        df_test_id.columns = [col.replace('"', '') for col in df_test_id.columns]
        df_test_id.columns = [col.replace(' ', '') for col in df_test_id.columns]
        file_list = [filename for filename in os.listdir(STRING.reporting_output) if filename.endswith('.csv')]
        df_bl = pd.read_csv(STRING.reporting_output + file_list[0], sep=';', encoding='utf-8')
        df_bl = df_bl.drop_duplicates(subset=['id_siniestro', 'nif_o_intm'], keep='last')
        df_bl = df_bl[['nif_o_intm']]
        df_bl['nif_o_intm'] = df_bl['nif_o_intm'].map(str)
        df_bl['Count'] = df_bl.groupby('nif_o_intm')['nif_o_intm'].transform('count')
        df_bl = df_bl.drop_duplicates(subset=['nif_o_intm'])
        file_df = df_test_id[['id_siniestro', 'id_fiscal']]
        file_df['id_fiscal'] = file_df['id_fiscal'].map(str)
        file_df['id_siniestro'] = file_df['id_siniestro'].map(int)
        file_df = file_df[['id_siniestro', 'id_fiscal']]
        file_df = pd.merge(file_df, df_bl, how='left', left_on='id_fiscal', right_on='nif_o_intm')
        del file_df['nif_o_intm']
        del file_df['id_fiscal']
        file_df['Count'] = file_df['Count'].fillna(0)
        file_df.columns = ['id_siniestro', 'checklist2']
        return file_df

    @staticmethod
    def checklist3():
        """Cambio reciente de coberturas y declaración inmediata de siniestro que afecta al cambio 
        (p.ej. aumento de capitales)
        :return: This return a Dataframe with the columns 'id_siniestro', 'checklist3i', where 'checklist3a' is the 
        difference between the last guarantee modification and the claim occurance, 'checklist3b) is the difference
        between the last data modification and the claim occurance, checklist3c) is the difference between the 
        last SUPLEMENTO added and the claim occurance.
        """
        df_test_hist_mov_pol_ref = pd.read_csv(
            STRING.histmovpolref_input_prediction, sep=',', encoding='latin1', quotechar='"')

        df_test_hist_mov_pol_ref = df_test_hist_mov_pol_ref[['id_siniestro', 'hist_movimiento_mod_garantias_fecha',
                                                             'hist_movimiento_siniestro_fecha',
                                                             'hist_movimiento_suplemento_fecha',
                                                             'hist_movimiento_mod_datos_fecha']]

        for i in ['hist_movimiento_mod_garantias_fecha', 'hist_movimiento_siniestro_fecha',
                  'hist_movimiento_suplemento_fecha', 'hist_movimiento_mod_datos_fecha']:
            df_test_hist_mov_pol_ref[i] = pd.to_datetime(df_test_hist_mov_pol_ref[i], format='%Y-%m-%d',
                                                         errors='coerce')

        df_test_hist_mov_pol_ref['checklist3a'] = pd.Series(df_test_hist_mov_pol_ref['hist_movimiento_siniestro_fecha']
                                                            - df_test_hist_mov_pol_ref[
                                                                'hist_movimiento_mod_garantias_fecha']).dt.days / 365

        df_test_hist_mov_pol_ref['checklist3b'] = pd.Series(df_test_hist_mov_pol_ref['hist_movimiento_siniestro_fecha']
                                                            - df_test_hist_mov_pol_ref[
                                                                'hist_movimiento_mod_datos_fecha']).dt.days / 365

        df_test_hist_mov_pol_ref['checklist3c'] = pd.Series(df_test_hist_mov_pol_ref['hist_movimiento_siniestro_fecha']
                                                            - df_test_hist_mov_pol_ref[
                                                                'hist_movimiento_suplemento_fecha']).dt.days / 365

        del df_test_hist_mov_pol_ref['hist_movimiento_mod_garantias_fecha']
        del df_test_hist_mov_pol_ref['hist_movimiento_mod_datos_fecha']
        del df_test_hist_mov_pol_ref['hist_movimiento_suplemento_fecha']
        del df_test_hist_mov_pol_ref['hist_movimiento_siniestro_fecha']

        return df_test_hist_mov_pol_ref

    @staticmethod
    def checklist4():
        """
        2 ó + siniestros con daños importantes (> 5.000 euros en Hogar y Comunidades ó >10.000 en Comercio) (*)
        # Se toma la reserva inicial del siniestro.
        :return: This return a Dataframe with the columns 'id_siniestro', 'checklist4_poliza', 'checklist4_nif', where 
        'checklist4_' represents how many sinister (by policy/nif) has an initial reserve >= 5000 since 2015
        """
        df_test_id = pd.read_csv(STRING.id_input_prediction, sep=',',
                                 encoding='utf-8', quotechar='"')
        df_test_id.columns = [col.replace('"', '') for col in df_test_id.columns]
        df_test_id.columns = [col.replace(' ', '') for col in df_test_id.columns]
        df_po_reserva_test = pd.read_csv(STRING.poreservable_input_prediction,
                                         sep=',',
                                         encoding='utf-8', quotechar='"')
        reserva_base = pd.read_csv(STRING.poreservable_input_training,
                                   sep=',',
                                   encoding='utf-8', quotechar='"')
        id_base = pd.read_csv(STRING.id_input_training,
                              sep=',',
                              encoding='utf-8', quotechar='"')
        id_base.columns = [col.replace('"', '') for col in id_base.columns]
        id_base.columns = [col.replace(' ', '') for col in id_base.columns]
        id_base = id_base[['id_siniestro', 'id_fiscal']]
        df_test_id = df_test_id[['id_siniestro', 'id_fiscal']]
        id_base = pd.concat([id_base, df_test_id], axis=0)

        # We take the variables we need and concat the new sinister with past sinister
        reserva_indem_base = reserva_base[['id_siniestro', 'po_res_cobertura_id', 'po_res_indem', 'id_poliza']]
        reserva_indem = df_po_reserva_test[['id_siniestro', 'po_res_cobertura_id', 'po_res_indem', 'id_poliza']]

        reserva_indem = pd.concat([reserva_indem, reserva_indem_base], axis=0)
        del reserva_base
        del reserva_indem_base

        # We merge with ID by sinister
        reserva_indem['id_siniestro'] = reserva_indem['id_siniestro'].map(int)
        id_base['id_siniestro'] = id_base['id_siniestro'].map(int)

        reserva_indem = pd.merge(reserva_indem, id_base, how='left', on='id_siniestro')

        # We calculate the initial RESERVA for each policy and create the variable RESERVA > 5000
        reserva_indem = reserva_indem.drop_duplicates(subset=['id_siniestro', 'po_res_cobertura_id',
                                                              'po_res_indem'],
                                                      keep='first')
        del reserva_indem['po_res_cobertura_id']
        reserva_indem['po_res_indem'] = reserva_indem['po_res_indem'].map(float)
        reserva_indem = reserva_indem.groupby(['id_siniestro', 'id_poliza', 'id_fiscal'])[
            'po_res_indem'].sum().reset_index()
        reserva_indem['po_res_indem_mayor_5000'] = pd.Series(0, index=reserva_indem.index)
        reserva_indem.loc[reserva_indem['po_res_indem'] >= 5000, 'po_res_indem_mayor_5000'] = 1

        # Now we have the values by sinister, we group by id_poliza and by nif
        poliza_indem = reserva_indem.groupby(['id_poliza'])['po_res_indem_mayor_5000'].sum().reset_index()
        nif_indem = reserva_indem.groupby(['id_fiscal'])['po_res_indem_mayor_5000'].sum().reset_index()

        # Merge the results
        poliza_indem['id_poliza'] = poliza_indem['id_poliza'].map(str)
        reserva_indem['id_poliza'] = reserva_indem['id_poliza'].map(str)
        poliza_indem.columns = ['id_poliza', 'checklist4_poliza']

        nif_indem['id_fiscal'] = nif_indem['id_fiscal'].map(str)
        reserva_indem['id_fiscal'] = reserva_indem['id_fiscal'].map(str)
        nif_indem.columns = ['id_fiscal', 'checklist4_nif']

        reserva_indem = pd.merge(reserva_indem, poliza_indem, how='left', on='id_poliza')
        reserva_indem = pd.merge(reserva_indem, nif_indem, how='left', on='id_fiscal')
        del reserva_indem['id_poliza']
        del reserva_indem['id_fiscal']

        # We need just to take the new sinister
        df_po_reserva_test = df_po_reserva_test[['id_siniestro']]
        df_po_reserva_test['id_siniestro'] = df_po_reserva_test['id_siniestro'].map(int)
        df_po_reserva_test = pd.merge(df_po_reserva_test, reserva_indem, how='left', on='id_siniestro')

        return df_po_reserva_test
