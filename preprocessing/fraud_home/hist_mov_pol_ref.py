from pyspark.sql.functions import when, udf, lit, datediff, year, month, \
    dayofmonth, date_format
from pyspark.sql.types import IntegerType, DateType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import functions as f, outliers


class HistMovPolRef(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Historico Movimiento Poliza Referencia")

    def run(self):
        df, df_base = self._extract_data()
        df = self._transform_data(df, df_base)
        self._load_data(df)

    def _extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """
        if self._is_diario:
            df = (
                self._spark
                    .read
                    .csv(STRING.histmovpolref_input_prediction, header=True, sep=',', nullValue='?'))

            df_base = (
                    self._spark
                        .read
                        .csv(STRING.histmovpolref_input_training, header=True, sep=',', nullValue='?'))
        else:
            df = (
                self._spark
                    .read
                    .csv(STRING.histmovpolref_input_training, header=True, sep=',', nullValue='?'))

            df_base = None

        return df, df_base

    def _transform_data(self, df, df_base):
        """Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # Creamos Dummies para los últimos tipos de movimiento
        types = df.select('hist_movimiento_tipo_movimiento').distinct().collect()
        types = [ty['hist_movimiento_tipo_movimiento'] for ty in types]
        types_list = [when(df['hist_movimiento_tipo_movimiento'] == ty, 1).otherwise(0).alias(
            'd_hist_movimiento_tipo_movimiento_' + ty) for ty in types if ty is not None]
        df = df.select(list(df.columns) + types_list)
        df = df.drop('hist_movimiento_tipo_movimiento')

        # 6) Creamos dummies para el motivo de movimiento
        motivos_importantes = [102, 109, 165, 284, 375, 3117, 101,
                               104, 105, 106, 109, 114, 117, 129,
                               133, 165, 166, 218, 370, 375, 376, 378, 528, 702, 703, 705, 707, 651, 119, 320, 803, 789,
                               790, 791, 792, 793, 794, 795, 796, 802, 655, 656, 657, 658, 659, 660, 661, 665, 666, 651,
                               652, 378, 320, 324, 325, 259, 223, 138, 139, 133
                               ]
        df = df.withColumn('hist_movimiento_motivo_movimiento',
                           df['hist_movimiento_motivo_movimiento'].cast(IntegerType()))
        df = df.withColumn('hist_movimiento_motivo_sospechoso',
                           when(df['hist_movimiento_motivo_movimiento'].isin(motivos_importantes), 1).otherwise(0))

        # Calculamos para las fechas, año, mes, día, día-semana para las siguientes variables:
        var_fecha = ["hist_movimiento_suplemento_fecha",
                     "hist_movimiento_mod_datos_fecha", "hist_movimiento_mod_garantias_fecha",
                     "hist_movimiento_fecha_inicio"]

        func = udf(lambda x: f.replace_date(x), DateType())
        for col in var_fecha:
            year_name = str(col) + '_year'
            month_name = str(col) + '_month'
            day_name = str(col) + '_day'
            weekday_name = str(col) + '_weekday'
            df = df.withColumn(col, when(df[col].isin(['0', 0]), None).otherwise(df[col]))
            df = df.fillna({col: '1900/01/01'})
            df = df.withColumn(col, func(df[col]))
            df = df.withColumn(col, when(df[col] == '1900-01-01', None).otherwise(df[col]))
            df = df.withColumn(year_name, year(df[col]))
            df = df.withColumn(month_name, month(df[col]))
            df = df.withColumn(day_name, dayofmonth(df[col]))
            df = df.withColumn(weekday_name, date_format(col, 'u') - 1)  # We adapt to (0=Monday, 1=Tuesday...)
            df = df.withColumn(weekday_name, df[weekday_name].cast(IntegerType()))

        df = df.withColumn('hist_movimiento_siniestro_fecha', func(df['hist_movimiento_siniestro_fecha']))
        df = df.withColumn('hist_movimiento_vto_natural', when(df['hist_movimiento_vto_natural'].isin(['0', 0]),
                                                               None).otherwise(df['hist_movimiento_vto_natural']))
        df = df.fillna({'hist_movimiento_vto_natural': '1900/01/01'})
        df = df.withColumn('hist_movimiento_vto_natural', func(df['hist_movimiento_vto_natural']))

        # 8 Fechas lógicas
        # Diferencia de fechas y modificaciones
        df = df.withColumn('fecha_diferencia_siniestro_suplemento',
                           datediff('hist_movimiento_siniestro_fecha', 'hist_movimiento_suplemento_fecha'))
        df = df.withColumn('fecha_diferencia_siniestro_datos',
                           datediff('hist_movimiento_siniestro_fecha', 'hist_movimiento_mod_datos_fecha'))
        df = df.withColumn('fecha_diferencia_siniestro_garantias',
                           datediff('hist_movimiento_siniestro_fecha', 'hist_movimiento_mod_garantias_fecha'))

        # Promedio de la duración de las versiones
        df = df.withColumn('fecha_duracion_version', df['hist_movimiento_tiempo_poliza'].cast(IntegerType()) / (
                    df['hist_movimiento_version'].cast(IntegerType()) + 1))

        # Fecha de vencimiento e inicio
        df = df.withColumn('fecha_vencimiento_siniestro',
                           datediff('hist_movimiento_vto_natural', 'hist_movimiento_siniestro_fecha'))
        df = df.withColumn('fecha_vencimiento_siniestro', when(df['fecha_vencimiento_siniestro'] < 0, 1).otherwise(0))
        df = df.withColumn('fecha_vencimiento_siniestro_15dias',
                           when(df['fecha_vencimiento_siniestro'] < 15, 1).otherwise(0))
        df = df.withColumn('fecha_vencimiento_siniestro_30dias',
                           when(df['fecha_vencimiento_siniestro'] < 30, 1).otherwise(0))

        # Outliers
        outliers_var = ['hist_movimiento_nro_suplemento', 'hist_movimiento_mod_datos', 'hist_movimiento_mod_garantias']
        if self._is_diario:
            df_base = df_base.select(*outliers_var)
            for col in outliers_var:
                df = outliers.Outliers.outliers_test_values(df, df_base, col, not_count_zero=True)
        else:
            for col in outliers_var:
                df = outliers.Outliers.outliers_mad(df, col, not_count_zero=True)

        fechas_nan = ['fecha_diferencia_siniestro_suplemento', 'fecha_diferencia_siniestro_datos',
                      'fecha_diferencia_siniestro_garantias']

        for col in fechas_nan:
            nan_name = str(col) + '_nan'
            max_value = df.agg({col: 'max'}).collect()[0][0] + 1
            df = df.withColumn(nan_name, when(df[col].isNull(), 1).otherwise(max_value))

        # Waird Days
        df = df.withColumn('hist_movimiento_suplemento_weekend',
                           when(df['hist_movimiento_suplemento_fecha_weekday'].isin([5, 6]), 1).otherwise(0))
        df = df.withColumn('hist_movimiento_datos_weekend',
                           when(df['hist_movimiento_suplemento_fecha_weekday'].isin([5, 6]), 1).otherwise(0))
        df = df.withColumn('hist_movimiento_garantias_weekend',
                           when(df['hist_movimiento_suplemento_fecha_weekday'].isin([5, 6]), 1).otherwise(0))

        delete_var = ['id_poliza', "hist_movimiento_version",
                      "hist_movimiento_estado", "hist_movimiento_tipo_movimiento",
                      "hist_movimiento_suplemento_fecha",
                      "hist_movimiento_mod_datos_fecha", "hist_movimiento_mod_garantias_fecha",
                      "hist_movimiento_siniestro_fecha",
                      "hist_movimiento_fecha_inicio", "hist_movimiento_fecha_efecto_natural",
                      "hist_movimiento_vto_natural", "hist_movimiento_tiempo_poliza",
                      'hist_movimiento_suplemento_fecha_year',
                      'hist_movimiento_suplemento_fecha_month',
                      'hist_movimiento_suplemento_fecha_day',
                      'hist_movimiento_suplemento_fecha_weekday',
                      'hist_movimiento_mod_datos_fecha_year',
                      'hist_movimiento_mod_datos_fecha_month',
                      'hist_movimiento_mod_datos_fecha_day',
                      'hist_movimiento_mod_datos_fecha_weekday',
                      'hist_movimiento_mod_garantias_fecha_year',
                      'hist_movimiento_mod_garantias_fecha_month',
                      'hist_movimiento_mod_garantias_fecha_day',
                      'hist_movimiento_mod_garantias_fecha_weekday',
                      'hist_movimiento_fecha_inicio_year',
                      'hist_movimiento_fecha_inicio_month',
                      'hist_movimiento_fecha_inicio_day',
                      'hist_movimiento_fecha_inicio_weekday', 'audit_siniestro_producto_tecnico',
                      'audit_siniestro_entidad_legal'
                      ]

        df = df.drop(*delete_var)
        df = df.fillna(-1)

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.histmovpolref_output_prediction
        else:
            name = STRING.histmovpolref_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main para test
if __name__ == '__main__':
    HistMovPolRef(is_diario=False).run()
