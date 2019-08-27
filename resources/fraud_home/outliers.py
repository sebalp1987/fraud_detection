import numpy as np

from pyspark.sql.functions import round as rnd, when, lit
from pyspark.sql.types import FloatType


class Outliers:

    def outliers_mad(file_df, col_name, not_count_zero=True):
        name = str(col_name) + '_mad_outlier'
        file_df_col = file_df.select(col_name).dropna(subset=col_name)
        if not_count_zero:
            file_df_col = file_df_col.filter(file_df_col[col_name] > 0)
        else:
            file_df_col = file_df_col.filter(file_df_col[col_name] >= 0)
        if file_df_col.count() > 1:
            # MAD
            outliers_mad = Outliers.mad_based_outlier(file_df_col)
            list_outlier = []
            points = np.array(file_df_col.select(col_name).collect())
            points = np.array(points).astype(float)

            for ax, func in zip(points, outliers_mad):
                if func:  # True is outlier
                    list_outlier.append(ax[0])
            list_outlier = [round(float(value), 2) for value in list_outlier]
            list_outlier = set(list_outlier)

            file_df_add = file_df.withColumn(name,
                                             when(rnd(file_df[col_name], 2).isin(list_outlier), 1).otherwise(0))
        else:
            file_df_add = file_df.withColumn(name, lit(0))
        return file_df_add

    def mad_based_outlier(points, thresh=3.5):
        points = np.array(points.collect()).astype(float)
        median = np.median(points, axis=0)

        diff = np.sum((points - median) ** 2, axis=-1)

        diff = np.sqrt(diff)

        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    def percentile_based_outlier(data, threshold=95):
        diff = (100 - threshold) / 2.0
        minval, maxval = np.percentile(data, [diff, 100 - diff])
        return (data < minval) | (data > maxval)

    def outliers_test_values(file_df, base_df, col_name, not_count_zero=True):
        name = str(col_name) + '_mad_outlier'
        # base_df
        base_df_col = base_df.select(col_name).dropna(subset=col_name)

        if not_count_zero:
            base_df_col = base_df_col.filter(base_df_col[col_name] > 0)
        else:
            base_df_col = base_df_col.filter(base_df_col[col_name] >= 0)

        # test df
        file_df_col = file_df.select(col_name).dropna(subset=col_name)
        file_df_col = file_df_col.withColumn(col_name, file_df_col[col_name].cast(FloatType()))

        if not_count_zero:
            file_df_col = file_df_col.filter(file_df_col[col_name] > 0)
        else:
            file_df_col = file_df_col.filter(file_df_col[col_name] >= 0)

        if file_df_col.count() > 1:
            # MAD
            median, med_abs_deviation = Outliers.mad_based_outlier_parameters(base_df_col)
            points = np.array(file_df_col.select(col_name).collect())
            points = np.array(points).astype(float)

            diff = np.sum((points - median) ** 2, axis=-1)
            diff = np.sqrt(diff)
            modified_z_score = 0.6745 * diff / med_abs_deviation

            outliers_mad = modified_z_score > 3.5
            list_outlier = []

            for ax, func in zip(points, outliers_mad):
                if func:  # True is outlier
                    list_outlier.append(ax[0])
            list_outlier = [round(float(value), 2) for value in list_outlier]
            list_outlier = set(list_outlier)

            file_df_add = file_df.withColumn(name, when(rnd(col_name, 2).isin(list_outlier), 1).otherwise(0))
        else:
            file_df_add = file_df.withColumn(name, lit(0))

        return file_df_add

    def mad_based_outlier_parameters(points):
        points = np.array(points.collect())
        points = np.array(points).astype(float)

        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)

        med_abs_deviation = np.median(diff)
        return median, med_abs_deviation
