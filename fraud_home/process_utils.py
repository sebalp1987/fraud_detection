import os

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from fraud_home.resources.fraud_home import STRING


class process_utils:

    def append_df(self: pd.DataFrame, df_add: pd.DataFrame, on_var: str = 'id_siniestro', on_var_type=int,
                  how='left'):
        """
        It appends a dataframe based on 'id_siniestro' using join left. Also it returns the column names of the new
        dataframe so in the next step we can evaluate missing values
        
        :param df_add: The new DataFrame
        :param on_var: The key column
        :param on_var_type: The type of the key column
        :return:  df_base + df_add, df_add column names
        """
        self[on_var] = self[on_var].map(on_var_type)
        df_add[on_var] = df_add[on_var].map(on_var_type)
        base_columns = self.columns.values.tolist()
        base_columns.remove(on_var)
        cols_to_use = df_add.columns.difference(base_columns)

        df_base = pd.merge(self, df_add[cols_to_use], how=how, on=on_var)
        df_add_cols = df_add.columns.values.tolist()
        df_add_cols.remove(on_var)
        print('final shape ', df_base.shape)
        return df_base, df_add_cols

    def variance_threshold(self: pd.DataFrame, cp, fecha, threshold=0.0):
        """
        VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance
        doesn’t meet some threshold. By default, it removes all zero-variance features, i.e.
        features that have the same value in all samples.
        As an example, suppose that we have a dataset with boolean features,
        and we want to remove all features that are either one or zero (on or off) in more than 80% of the samples.
        """
        column_names = self.columns.values.tolist()
        key_variables = ['id_siniestro', 'id_poliza', 'cod_filiacion'] + cp + fecha
        removed_var = []
        for i in key_variables:
            try:
                column_names.remove(i)
                removed_var.append(i)
            except:
                pass

        append_names = []
        for i in column_names:
            self_i = self[[i]]
            self_i = self_i.apply(pd.to_numeric, errors='coerce')
            self_i = self_i.dropna(how='any', axis=0)
            selection = VarianceThreshold(threshold=threshold)
            try:
                selection.fit(self_i)
                features = selection.get_support(indices=True)
                features = self_i.columns[features]
                features = [column for column in self_i[features]]
                selection = pd.DataFrame(selection.transform(self_i), index=self_i.index)
                selection.columns = features
                append_names.append(selection.columns.values.tolist())
            except:
                pass

        append_names = [item for sublist in append_names for item in sublist]
        append_names = list(set(append_names))
        self = self[removed_var + append_names]
        return self

    def fillna_by_bottle(self: pd.DataFrame, df_add_cols: list, cp, fecha, fill_value=-1):
        """
        Using append_df, we get the column names added. Then we will fillna only if the whole columns are NaN
        after they have been appended.
        
        :param df_add_cols: the column names just added
        :param fill_value: The fillna value we choose
        :param cp
        :param fecha
        :return: Dataframe with the fillna process
        """

        key_variables = cp + fecha + ['id_fiscal', 'id_poliza', 'cliente_codfiliacion']
        removed_var = []
        for i in key_variables:
            if i in self:
                removed_var.append(i)

        df_removed = self[removed_var]

        for i in key_variables:
            if i in self:
                del self[i]
            if i in df_add_cols:
                df_add_cols.remove(i)

        condition_nan_values = len(df_add_cols)

        for i in df_add_cols:
            self[i] = pd.to_numeric(self[i], errors='coerce')

        # We create a variable that count how many NAN values are in the selected columns df_add_cols
        self['count_NAN'] = self[df_add_cols].isnull().sum(axis=1)
        # We make a fillna only if 'count_NAN' is exactly the number of new columns

        df_base = self.apply(lambda x: x.fillna(fill_value) if x['count_NAN'] == condition_nan_values
                             else x, axis=1)
        del df_base['count_NAN']

        df_base = pd.concat([df_base, df_removed], axis=1)
        return df_base

    def delete_row_df_by_name(self: pd.DataFrame, del_name: str = 'MIGRA'):
        """
        If the del_name is contained in a column name it will delete the row which is == 1 from the dataframe

        :param del_name: String we want to search for deleting porpose.
        :return: df_base without the rows found
        """

        col_names = self.columns.values.tolist()
        col_names = [f for f in col_names if del_name in f]

        for i in col_names:
            self = self[self[i] != 1]
            self = self.drop(i, axis=1)

        return self

    @staticmethod
    def fillna_multioutput(self: pd.DataFrame, not_consider: ['id_siniestro', 'FRAUDE'], n_estimator=500,
                           max_depth=None, n_features=3):
        """
        Multioutput regression used for estimating NaN values columns. 
        :return: df with multioutput fillna
        """
        # First we determine which columns have NaN values are which not

        jcols = self.columns[self.isnull().any()].tolist()
        icols = self.columns.values.tolist()
        for i in jcols:
            icols.remove(i)

        # We evaluate here which rows are null value. This returns a boolean for each row
        notnans = self[jcols].notnull().all(axis=1)

        # We create a df with not nan values which will be used as train-test in a supervised model
        df_notnans = self[notnans]
        # We create a train-test set with X = icols values that do not have null values. And we try to estimate
        # the values of jcols (the columns with NaN). Here we are not considering the NaN values so we can estimate
        # as a supervised model the nan_cols. And finally, we apply the model estimation to the real NaN values.
        X_train, X_test, y_train, y_test = train_test_split(df_notnans[icols], df_notnans[jcols],
                                                            train_size=0.70,
                                                            random_state=42)

        n_estimator = n_estimator
        max_features = (round((len(df_notnans.columns)) / n_features))
        min_samples_leaf = round(len(df_notnans.index) * 0.005)
        if min_samples_leaf < 5:
            min_samples_leaf = 10
        min_samples_split = min_samples_leaf * 10
        max_depth = max_depth

        print('RANDOM FOREST WITH: ne_estimator=' + str(n_estimator) + ', max_features=' + str(max_features) +
              ', min_samples_leaf=' + str(min_samples_leaf) + ', min_samples_split='
              + str(min_samples_split) + ', max_depth=' + str(max_depth))

        regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth,
                                                                  random_state=42, verbose=1,
                                                                  max_features=max_features,
                                                                  min_samples_split=min_samples_split,
                                                                  min_samples_leaf=min_samples_leaf))

        # We fit the model deleting variables that must not be included to do not have endogeneity (for example FRAUD
        # variable)
        regr_multirf.fit(X_train.drop(not_consider, axis=1), y_train)

        # We get R2 to determine how well is explaining the model
        score = regr_multirf.score(X_test.drop(not_consider, axis=1), y_test)
        print('R2 model ', score)

        # Now we bring the complete column dataframe with NaN row values
        df_nans = self.loc[~notnans].copy()
        df_not_nans = self.loc[notnans].copy()
        # Finally what we have to do is to estimate the NaN columns from the previous dataframe. For that we use
        # multioutput regression. This will estimate each specific column using Random Forest model. Basically we
        # need to predict dataframe column NaN values for each row in function of dataframe column not NaN values.
        df_nans[jcols] = regr_multirf.predict(df_nans[icols].drop(not_consider, axis=1))

        df_without_nans = pd.concat([df_nans, df_not_nans], axis=0, ignore_index=True)

        df = pd.merge(self, df_without_nans, how='left', on='id_siniestro', suffixes=('', '_y'))

        for i in jcols:
            df[i] = df[i].fillna(df[i + '_y'])
            del df[i + '_y']

        filter_col = [col for col in df if col.endswith('_y')]
        for i in filter_col:
            del df[i]

        return df

    @staticmethod
    def robust_scale(self: pd.DataFrame, quantile_range=(25.0, 75.0)):
        """
        Scale features using statistics that are robust to outliers.
        This Scaler removes the median and scales the data according to the quantile range 
        (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) 
        and the 3rd quartile (75th quantile).
        :return: scaled df
        """

        robust_scaler = RobustScaler(quantile_range=quantile_range)

        df_cols = self.columns.values.tolist()
        df_cols.remove('id_siniestro')

        params = []

        for i in df_cols:
            X = self[[i]]
            self[i] = robust_scaler.fit_transform(X)
            center = robust_scaler.center_
            scale = robust_scaler.scale_
            center = float(center[0])
            scale = float(scale[0])
            params.append([str(i), center, scale])

        return self, params

    @staticmethod
    def pca_reduction(self: pd.DataFrame, variance=95.00):
        """
        This automatically calcualte a PCA to df taking into account the 95% of the dataset explained variance
        :param show_plot: Threshold Variance Plot
        :param variance: Dataset variance limit to consider in the PCA.
        :return: PCA df
        """

        siniestro_df = self[['id_siniestro']]
        del self['id_siniestro']

        columns = len(self.columns)
        scaler = StandardScaler()
        X = scaler.fit_transform(self)
        pca = PCA(whiten=True, svd_solver='randomized', n_components=columns)

        pca.fit(X)
        cumsum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
        cumsum = list(cumsum)
        var = [value for value in cumsum if value <= variance]
        pca_components = len(var)

        pca = PCA(n_components=pca_components, whiten=True, svd_solver='randomized')
        params_scale = []
        for i in self.columns.values.tolist():
            X_i = self[[i]]
            self[i] = scaler.fit_transform(X_i)
            mean = scaler.mean_
            var = scaler.var_
            mean = float(mean[0])
            var = float(var[0])
            params_scale.append([str(i), mean, var])

        pca_components = pca.fit(self)
        df = pd.DataFrame(pca_components.transform(self))
        df = pd.concat([df, siniestro_df], axis=1)

        return df, pca_components, params_scale

    @staticmethod
    def append_blacklist(self: pd.DataFrame):
        """
        It append the variable FRAUDE = 1 in the dataframe passed.
        :return: df + FRAUDE
        """
        file_list = [filename for filename in os.listdir(STRING.reporting_output) if filename.endswith('.csv')]
        df_bl_resume = pd.read_csv(STRING.reporting_output + file_list[0], sep=';', encoding='utf-8')
        df_bl_resume['id_siniestro'] = df_bl_resume['id_siniestro'].map(int)
        df_bl_resume = df_bl_resume[['id_siniestro']]
        df_bl_resume['FRAUDE'] = pd.Series(1, index=df_bl_resume.index)
        df_bl_resume = df_bl_resume.drop_duplicates(subset='id_siniestro')
        self['id_siniestro'] = self['id_siniestro'].map(int)
        df = pd.merge(self, df_bl_resume, how='left', on='id_siniestro')
        df['FRAUDE'] = df['FRAUDE'].fillna(0)
        del df_bl_resume

        return df

    @staticmethod
    def output_normal_anormal_new(self: pd.DataFrame, output_file=True, input_name_file='raw_file', feedback=None):
        """
        It split the dataframe into three dataframes (normal, anormal, new) based on FRAUDE = (0,1) and New Sinister 
        Bottle. Also, if 'output_file' = True, it creates a new version of the final table.
        :param output_file: Boolean if it is necessary an output file
        :param input_name_file: The name of the output file.
        :param feedback: If True it considers the feedback of the IO.
        :return: Two dataframes based on normally anormally.
        """

        # First we separete New sinister
        new = self[self['TEST'] == 1]
        df = self[self['TEST'] == 0]
        df['FRAUDE'] = df['FRAUDE'].map(int)

        # Check feedback
        if feedback is not None:
            feedback = feedback.rename(columns={'Nº SINIESTRO': 'id_siniestro', 'RESULTADO': 'resultado'})
            feedback = feedback[['id_siniestro', 'resultado']]
            feedback['resultado'] = feedback['resultado'].str.upper()
            feedback = feedback.drop_duplicates()
            feedback = feedback.sort_values(by=['resultado'], ascending=[False])
            feedback = feedback.drop_duplicates(subset=['id_siniestro'], keep='first')
            feedback = feedback.dropna(subset=['id_siniestro'])

            # First, we keep only POSITIVOS and NEGATIVOS in feedback
            feedback['id_siniestro'] = feedback['id_siniestro'].map(int)
            feedback = feedback[feedback['resultado'] != 'POSIBLE FRAUDE TARDIO']
            feedback['FRAUDE_feed'] = pd.Series(-1, index=feedback.index)
            feedback = feedback.dropna(subset=['resultado'])
            feedback.loc[feedback['resultado'].str.startswith('POS'), 'FRAUDE_feed'] = 1
            # feedback.loc[feedback['resultado'] == 'POSIBLE FRAUDE TARDIO', 'FRAUDE_feed'] = 1
            feedback.loc[feedback['resultado'].str.startswith('NEG'), 'FRAUDE_feed'] = 0
            feedback = feedback[feedback['FRAUDE_feed'] >= 0]
            del feedback['resultado']

            # BASE (Keep FRAUDE = 1, FEEDBACK=1, FRAUDE=0)
            # Replace training values in closed sinister
            df['id_siniestro'] = df['id_siniestro'].map(int)
            df = pd.merge(df, feedback, how='left', on='id_siniestro')

            # Keep FRAUDE_feedback = 0 outside the Unsupervised, because we know that they are negatives
            base_feedback_0 = df[df['FRAUDE_feed'] == 0]
            df = df[df['FRAUDE_feed'] != 0]

            # Keep FRAUDE = 1, FEEDBACK = 1
            df.loc[df['FRAUDE_feed'] == 1, 'FRAUDE'] = 1

            # NEW (Keep FEEDBACK = 1) 
            # Incorporate this cases as BASE
            new['id_siniestro'] = new['id_siniestro'].map(int)
            new = pd.merge(new, feedback, how='left', on='id_siniestro')
            new['FRAUDE_feed'] = new['FRAUDE_feed'].fillna(-1)

            # Keep FEEDBACK=1 as part of base_feed
            base_feed = new[new['FRAUDE_feed'] == 1]

            # Keep FEEDBACK=0 as part of the Base but outside the Unsupervised
            test_feedback_0 = new[new['FRAUDE_feed'] == 0]
            feedback_0 = pd.concat([base_feedback_0, test_feedback_0], axis=0)

            # We clean the actual New Claims
            new = new[new['FRAUDE_feed'] == -1]
            del new['FRAUDE_feed']

            # Assign the FRAUDE=1 variable to the new feedback to the BASE
            base_feed['FRAUDE'] = pd.Series(1, index=base_feed.index)

            # Assing the FRAUDE=0 variable to the new feedback incorporated Supervised
            feedback_0['FRAUDE'] = pd.Series(0, index=feedback_0.index)
            del feedback_0['FRAUDE_feed']
            feedback = feedback_0

            # Pass to the detabase the closed cases
            df = pd.concat([df, base_feed], axis=0)
            del df['FRAUDE_feed']

        del new['TEST']
        del df['TEST']
        del feedback_0['TEST']

        anomaly = df[df['FRAUDE'] == 1]
        normal = df[df['FRAUDE'] == 0]

        normal = normal.drop_duplicates(subset='id_siniestro')
        anomaly = anomaly.drop_duplicates(subset='id_siniestro')
        new = new.drop_duplicates(subset='id_siniestro')
        string_anomaly = 'anomaly_' + input_name_file
        string_normal = 'normal_' + input_name_file

        if output_file:
            path = 'batch_files\\'

            normal_file = path + string_normal + '.csv'
            anormal_file = path + string_anomaly + '.csv'
            new_file = path + 'new_sinister.csv'

            anomaly.to_csv(anormal_file, sep=';', index=False)
            normal.to_csv(normal_file, sep=';', index=False)
            new.to_csv(new_file, sep=';', index=False)
            if feedback is not None:
                feedback.to_csv(path + 'feedback_0.csv', sep=';', encoding='latin1', index=False)
        return normal, anomaly, new, feedback
