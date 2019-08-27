from datetime import date

from pyspark.sql.functions import when, lit
from pyspark.sql.types import IntegerType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home.outliers import Outliers


class Hogar(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Hogar")

    def run(self):
        # self.logger.info("Running Hogar")
        print("Running Hogar")
        df, hogar_base = self._extract_data()
        df = self._transform_data(df, hogar_base)
        self._load_data(df)
        self._spark.stop()

    def _extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """
        if self._is_diario:
            df = (
                self._spark
                    .read
                    .csv(STRING.hogar_input_prediction, header=True, sep=',', nullValue='?'))

            hogar_base = (
                self._spark
                        .read
                        .csv(STRING.hogar_input_training, header=True, sep=',', nullValue='?',
                            encoding='UTF-8'))
        else:
            df = (
                self._spark
                    .read
                    .csv(STRING.hogar_input_training, header=True, sep=',', nullValue='?'))

            hogar_base = None

        return df, hogar_base

    def _transform_data(self, df, hogar_base):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param hogar_base: Historical data
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # Correct Anio Construccion by Date
        year_today = date.today().year
        df = df.withColumn('hogar_anio_construccion', when(df['hogar_anio_construccion'] > year_today, None).otherwise(
            df['hogar_anio_construccion']))

        # Variables No Identificadas
        no_identify = ['hogar_capital_continente', 'hogar_capital_contenido', 'hogar_m2', 'hogar_anio_construccion',
                       'hogar_ubicacion']

        for col in no_identify:
            df = df.withColumn(col, when(df[col].isin(['0', 0]), None).otherwise(df[col]))
            df = df.withColumn('d_' + col + '_no_identificado', when(df[col].isNull(), 1).otherwise(0))

        # Hogar Variables: Replace BANC SABADELL values by Zurich values
        df = df.withColumn('hogar_caracter',
                           when(df['hogar_caracter'].like('%1%'), 'P').otherwise(df['hogar_caracter']))
        df = df.withColumn('hogar_caracter',
                           when(df['hogar_caracter'].like('%2%'), 'P').otherwise(df['hogar_caracter']))
        df = df.withColumn('hogar_caracter',
                           when(df['hogar_caracter'].like('%3%'), 'I').otherwise(df['hogar_caracter']))
        df = df.withColumn('hogar_caracter',
                           when(df['hogar_caracter'].like('%4%'), 'I').otherwise(df['hogar_caracter']))
        df = df.withColumn('hogar_caracter',
                           when(df['hogar_caracter'].like('%5%'), 'P').otherwise(df['hogar_caracter']))
        df = df.withColumn('hogar_caracter',
                           when(df['hogar_caracter'].like('%6%'), 'No Identificado').otherwise(df['hogar_caracter']))

        df = df.withColumn('hogar_tipo_vivienda_code', lit('no_identificado'))
        df = df.withColumn('hogar_tipo_vivienda_code',
                           when(df['hogar_tipo_vivienda'].like('%1%'), 'UF').otherwise(df['hogar_tipo_vivienda_code']))
        df = df.withColumn('hogar_tipo_vivienda_code',
                           when(df['hogar_tipo_vivienda'].like('%2%'), 'UF').otherwise(df['hogar_tipo_vivienda_code']))
        df = df.withColumn('hogar_tipo_vivienda_code',
                           when(df['hogar_tipo_vivienda'].like('%3%'), 'AT').otherwise(df['hogar_tipo_vivienda_code']))
        df = df.withColumn('hogar_tipo_vivienda_code',
                           when(df['hogar_tipo_vivienda'].like('%4%'), 'PB').otherwise(df['hogar_tipo_vivienda_code']))
        df = df.withColumn('hogar_tipo_vivienda_code',
                           when(df['hogar_tipo_vivienda'].like('%5%'), 'PI').otherwise(df['hogar_tipo_vivienda_code']))
        df = df.drop('hogar_tipo_vivienda')
        df = df.withColumnRenamed('hogar_tipo_vivienda_code', 'hogar_tipo_vivienda')

        df = df.withColumn('hogar_uso', when(df['hogar_uso'].like('%1%'), 'P').otherwise(df['hogar_uso']))
        df = df.withColumn('hogar_uso', when(df['hogar_uso'].like('%2%'), 'S').otherwise(df['hogar_uso']))
        df = df.withColumn('hogar_uso', when(df['hogar_uso'].like('%3%'), 'No Identificado').otherwise(df['hogar_uso']))

        # Hogar Dummy Variables
        hogar_dummies = ['hogar_tipo_vivienda', 'hogar_caracter', 'hogar_uso']
        for col in hogar_dummies:
            df = df.fillna({col: 'no_identificado'})
            types = df.select(col).distinct().collect()
            types = [ty[col] for ty in types]
            hogar_type = [when(df[col] == ty, 1).otherwise(0).alias('d_' + col + '_' + ty) for ty in types]
            df = df.select(list(df.columns) + hogar_type)

        # Numero de Seguridades
        df = df.withColumn('hogar_seguridad_null', when(df['hogar_numero_seguridad'] == 0, 1).otherwise(0))
        df = df.withColumn('hogar_seguridad_baja', when(df['hogar_numero_seguridad'] == 1, 1).otherwise(0))
        df = df.withColumn('hogar_seguridad_media', when(df['hogar_numero_seguridad'].isin([2, 3, 4]), 1).otherwise(0))
        df = df.withColumn('hogar_seguridad_alta', when(df['hogar_numero_seguridad'].isin([5, 6]), 1).otherwise(0))

        # CARGA SINIESTRAL
        df = df.withColumnRenamed('coste_del_siniestro_por_rol', 'hogar_carga_siniestral')
        df = df.fillna({'hogar_carga_siniestral': 0})

        # By Codigo Unico

        df.registerTempTable('table')
        df = self._spark.sql(
            "SELECT *, ROUND(SUM(hogar_carga_siniestral) OVER(PARTITION BY hogar_carga_siniestral), 2) AS "
            "hogar_carga_siniestral_codigo FROM table ORDER BY hogar_codigo_unico")
        self._spark.sql("DROP TABLE IF EXISTS table")
        df = df.fillna({'hogar_carga_siniestral_codigo': 0})

        # Outliers Variables
        outliers_var = ['hogar_capital_continente', 'hogar_capital_contenido', 'hogar_m2', 'hogar_anio_construccion',
                        'hogar_carga_siniestral']

        # In this case we use double outliers to first clean the bad inputation values
        # OUTLIERS
        if self._is_diario:
            # First we clean outliers based on Historical Data
            hogar_base = hogar_base.withColumnRenamed('coste_del_siniestro_por_rol', 'hogar_carga_siniestral')
            hogar_base = hogar_base.select(*outliers_var)
            for col in outliers_var:
                if col != 'hogar_carga_siniestral':
                    # We count zero because we want to clean the bad inputation values
                    hogar_base = Outliers.outliers_mad(hogar_base, col, not_count_zero=False)

            # For carga siniestral null mean == 0, we dont need to use bad inputation
            hogar_base = hogar_base.fillna({'hogar_carga_siniestral': 0})

            # Then we use historical base to define Outliers
            for col in outliers_var:
                if col != 'hogar_carga_siniestral':
                    column_select = hogar_base.select([col, col + '_mad_outlier'])

                    # We filter the bad inputation values to use as base data
                    column_select = column_select.filter(column_select[col + '_mad_outlier'] == 0)
                    # Now we dont count zero values
                    df = Outliers.outliers_test_values(df, column_select, col, not_count_zero=True)

            column_select = hogar_base.select('hogar_carga_siniestral')
            df = df.fillna({'hogar_carga_siniestral': 0})
            df = Outliers.outliers_test_values(df, column_select, col, not_count_zero=True)

        else:
            for col in outliers_var:
                if col != 'hogar_carga_siniestral':
                    df = Outliers.outliers_mad(df, col, not_count_zero=False)  # Bad inputation clean
                    outlier_name = col + '_mad_outlier'
                    df = df.withColumn(col, when(df[outlier_name] == 1, None).otherwise(df[col]))
                    df = Outliers.outliers_mad(df, col, not_count_zero=True)
            df = df.fillna({'hogar_carga_siniestral': 0})
            df = Outliers.outliers_mad(df, 'hogar_carga_siniestral', not_count_zero=True)

        # DELETE VARIABLES
        del_variables = ['id_poliza', 'id_fiscal', 'hogar_tipo_via', 'hogar_nombre_via', 'hogar_numero_via',
                         'hogar_info_adicional',
                         'hogar_direccion_completa', 'hogar_codigo_unico',
                         'hogar_cod_poblacion', 'hogar_anio_construccion',
                         'poliza_producto_tecnico', 'poliza_producto_comercial', 'hogar_tipo_vivienda',
                         'hogar_poblacion', 'hogar_provincia', 'hogar_ubicacion', 'hogar_caracter',
                         'hogar_uso', 'version_poliza', 'auditFechaAperturaSiniestroReferencia',
                         'audit_poliza_entidad_legal'
                         ]
        df = df.drop(*del_variables)
        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.hogar_output_prediction
        else:
            name = STRING.hogar_output_training

        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


if __name__ == '__main__':
    Hogar(False).run()
