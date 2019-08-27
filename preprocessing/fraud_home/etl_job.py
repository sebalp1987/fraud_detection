import os
import pandas as pd

from fraud_home.configs import config

from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import process_utils

class FinalBottle:

    def __init__(self, is_diario):
        self._is_diario = is_diario

    def run(self):
        self._merge_data()

    def _merge_data(self):

        if self._is_diario:
            path_input = STRING.pipeline_preprocessing_prediction_output_path
            path_output = STRING.etl_diaria
            filter_name = 'prediction'
        else:
            path_input = STRING.pipeline_preprocessing_training_output_path
            path_output = STRING.etl_mensual
            filter_name = 'training'

        # We get the files without mediador and id
        files = set([f for f in os.listdir(path_input)])
        files = [f for f in files if filter_name in f and ('mediador' or 'id' or 'reporting' or '.csv') not in f]

        # Id is the base
        file_list = [filename for filename in os.listdir(path_input + 'id_preprocessed_' + filter_name) if
                     filename.endswith('.csv')]
        df = pd.read_csv(path_input + 'id_preprocessed_' + filter_name + '/' + file_list[0], sep=';',
                         dtype={'id_siniestro': int})

        # We remove date values and CP
        cp = config.parameters.get("cp")
        fecha_var = config.parameters.get("fecha_var")

        for file in files:
            file_list = [filename for filename in os.listdir(path_input + file) if
                         filename.endswith('.csv')]
            df_i = pd.read_csv(path_input + file + '/' + file_list[0], sep=';', dtype={'id_siniestro': int})
            print('Loading... ', file)
            print('shape ', df_i.shape)
            # Variance Threshold
            shape_0 = len(df_i.columns)
            print(shape_0)
            shape_1 = len(df_i.columns)
            print(shape_1)
            diff = shape_0 - shape_1
            print('Deleted Variables :', diff)
            # We append to the base (date to the left)
            print('Appending....')
            print('base shape ', df.shape)
            print(df.columns.values.tolist())
            print(df_i.columns.values.tolist())
            if 'fecha' not in file:
                df, df_add_cols = process_utils.process_utils.append_df(df, df_i, on_var='id_siniestro',
                                                                        on_var_type=int,
                                                                        how='left')
            else:
                df, df_add_cols = process_utils.process_utils.append_df(df, df_i, on_var='id_siniestro',
                                                                        on_var_type=int,
                                                                        how='right')
            print('new shape ', df.shape)
            print(df.columns.values.tolist())
            print(' ')
            # Fill values from merge (When a row from a bottle does not exist, e.g., historical data)
            print('Fillna Values... ')
            print('Total NaN Values Before ', df.isnull().sum().sum())
            df = process_utils.process_utils.fillna_by_bottle(df, df_add_cols, cp, fecha_var, fill_value=-1)
            print('Total NaN Values After ', df.isnull().sum().sum())
            print(' ')
            # Remove Migrated Rows
            print('Deleting Useless Rows...')
            print('row numbers Before', len(df.index))
            df = process_utils.process_utils.delete_row_df_by_name(df, del_name='MIGRA')
            print('row numbers After ', len(df.index))

        df.to_csv(path_output, sep=';', index=False)
        if self._is_diario:
            df.to_csv(STRING.etl_output_model, sep=';', index=False)

        return None


if __name__ == '__main__':
    FinalBottle(is_diario=False).run()
