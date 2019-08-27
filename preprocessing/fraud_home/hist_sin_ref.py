import datetime


from pyspark.sql.functions import when, udf, lit, upper, substring
from pyspark.sql.types import IntegerType, StringType, DateType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import functions as f


class HistSinRef(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Historico Siniestro Referencia")

    def run(self):
        df = self._extract_data()
        df = self._transform_data(df)
        self._load_data(df)

    def _extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """
        if self._is_diario:

            df = (
                self._spark
                    .read
                    .csv(STRING.histsinref_input_prediction, header=True, sep=',', nullValue='?'))
        else:

            df = (
                self._spark
                    .read
                    .csv(STRING.histsinref_input_training, header=True, sep=',', nullValue='?'))

        return df

    @staticmethod
    def _transform_data(df):
        """Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # Date Variables
        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())
        df = df.withColumn('hist_siniestro_actual_fecha', func(df['hist_siniestro_actual_fecha']))

        # USUARIO: If it is a manual user (if it is not AIDE, MIGRACION or BATCH = 1)
        df = df.withColumn('hist_siniestro_usuario_manual', when(
            df['hist_siniestro_actual_nombre_usuario_apertura'].isin(['USUARAIDE', 'MSD MIGRA', 'BATCH1',
                                                                      'BATCH2', 'SGR']), 1).otherwise(0))

        # USUARIO DE APERTURA: What kind of user is it. TCE, Zurich or Web Service.
        df = df.withColumn('hist_siniestro_usuario_cliente',
                           when(df['hist_siniestro_actual_nombre_usuario_apertura'].startswith('TCE'), 1).otherwise(0))
        df = df.withColumn('hist_siniestro_usuario_Z',
                           when(df['hist_siniestro_actual_nombre_usuario_apertura'].startswith('Z'), 1).otherwise(0))
        df = df.withColumn('hist_siniestro_usuario_WS',
                           when(df['hist_siniestro_actual_nombre_usuario_apertura'].startswith('WS'), 1).otherwise(0))

        # OFICINA APERTURA: We get dummies for each OFICINA APERUTRA
        df = df.fillna({'hist_siniestro_actual_oficina_apertura': 'No Informado'})

        df = df.withColumn('hist_siniestro_actual_oficina_apertura',
                           upper(df['hist_siniestro_actual_oficina_apertura']))

        replace_start_dict = {'ZURICH': 'ZURICH', 'CTD LINEAS PERSONALES': 'CTD LINEAS PERSONALES', 'CTD': 'CTD',
                              'CTA': 'CTA', 'CS': 'CSS', 'CENTRO': 'CSS'}
        replace_contain = {'MIGRACION': 'MIGRACION', ' RC ': 'CT RC'}

        funct = udf(lambda x: f.replace_dict_starswith(x, key_values=replace_start_dict), StringType())
        df = df.withColumn('hist_siniestro_actual_oficina_apertura',
                           funct(df['hist_siniestro_actual_oficina_apertura']))

        funct = udf(lambda x: f.replace_dict_contain(x, key_values=replace_contain), StringType())
        df = df.withColumn('hist_siniestro_actual_oficina_apertura',
                           funct(df['hist_siniestro_actual_oficina_apertura']))

        types_of_apertura = df.select('hist_siniestro_actual_oficina_apertura').distinct().collect()
        types_of_apertura = [col['hist_siniestro_actual_oficina_apertura'] for col in types_of_apertura]
        types_of_apertura_list = [when(df['hist_siniestro_actual_oficina_apertura'] == ty, 1).otherwise(0).alias(
            'd_hist_siniestro_actual_oficina_apertura_' + ty) for ty in types_of_apertura]

        df = df.select(list(df.columns) + types_of_apertura_list)
        df.drop('hist_siniestro_actual_oficina_apertura')

        # HORA APERTURA: We generate differente hour ranges
        df = df.withColumn('hist_siniestro_actual_hora_apertura',
                           substring('hist_siniestro_actual_hora_apertura', 0, 2).cast(IntegerType()))

        df = df.withColumn('hist_siniestro_actual_hora_nocturna',
                           when(df['hist_siniestro_actual_hora_apertura'].between(0, 6), 1).otherwise(0))

        df = df.withColumn('hist_siniestro_actual_hora_laboral',
                           when(df['hist_siniestro_actual_hora_apertura'].between(7, 18), 1).otherwise(0))

        df = df.withColumn('hist_siniestro_actual_hora_no_laboral',
                           when(df['hist_siniestro_actual_hora_apertura'].between(19, 23), 1).otherwise(0))

        df = df.withColumn('hist_siniestro_actual_hora_error_migra',
                           when(df['hist_siniestro_actual_hora_apertura'].isNull(), 1).otherwise(0))

        # DELETE VARIABLES
        delete_variables = ['id_poliza', 'version_poliza', "hist_siniestro_actual_fecha",
                            "hist_siniestro_actual_tipo_operacion",
                            "hist_siniestro_actual_nombre_usuario_apertura", "hist_siniestro_actual_hora_apertura",
                            "audit_siniestro_producto_tecnico",
                            "audit_siniestro_codigo_compania"
                            ]

        df = df.drop(*delete_variables)

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.histsinref_output_prediction
        else:
            name = STRING.histsinref_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main para test
if __name__ == '__main__':
    HistSinRef(is_diario=False).run()
