from pyspark.sql.functions import when, udf
from pyspark.sql.types import IntegerType, StringType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home.functions import replace_dict


class Siniestro(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Siniestro")

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
                    .csv(STRING.siniestro_input_prediction, header=True, sep=',', nullValue='?'))
        else:
            df = (
                self._spark
                    .read
                    .csv(STRING.siniestro_input_training, header=True, sep=',', nullValue='?'))

        return df

    @staticmethod
    def _transform_data(df):
        """Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # USUARIO: If it is a manual user (if it is not AIDE, MIGRACION or BATCH = 1)
        df = df.withColumn('siniestro_usuario_manual',
                           when(df['siniestro_nombre_usuario_apertura'].isin(['USUARAIDE', 'MSD MIGRA', 'BATCH1',
                                                                              'BATCH2', 'SGR']), 0).otherwise(1))

        # USUARIO DE APERTURA: What kind of user is it. TCE, Zurich or Web Service.
        df = df.withColumn('siniestro_usuario_cliente',
                           when(df['siniestro_nombre_usuario_apertura'].startswith('TCE'), 1).otherwise(0))
        df = df.withColumn('siniestro_usuario_Z',
                           when(df['siniestro_nombre_usuario_apertura'].startswith('Z'), 1).otherwise(0))
        df = df.withColumn('siniestro_usuario_WS',
                           when(df['siniestro_nombre_usuario_apertura'].startswith('WS'), 1).otherwise(0))

        # OFICINA APERTURA: DUMMIES for each OFICINA APERTURA
        df = df.fillna({'siniestro_oficina_apertura': 'No Informado'})

        key_values = {'ZURICH': 'ZURICH', 'CTD LINEAS PERSONALES': 'CTD LINEAS PERSONALES', 'MIGRACION': 'MIGRACION',
                      'CTD': 'CTD', 'CTA': 'CTA', 'CS': 'CSS', 'CENTRO': 'CSS', 'CT RC': 'CT RC', 'LESIONES': 'CT RC',
                      'BASICOS': 'CT RC'}

        funct = udf(lambda x: replace_dict(x, key_values=key_values, key_in_value=True), StringType())
        df = df.withColumn('siniestro_oficina_apertura', funct(df['siniestro_oficina_apertura']))

        type_oficina = df.select('siniestro_oficina_apertura').distinct().collect()
        type_oficina = [ty.siniestro_oficina_apertura for ty in type_oficina]

        var_type_oficina = [
            when(df['siniestro_oficina_apertura'] == ty, 1).otherwise(0).alias('d_siniestro_oficina_apertura_' + ty) for
            ty
            in type_oficina]
        df = df.select(list(df.columns) + var_type_oficina)
        df = df.drop('siniestro_oficina_apertura')

        # DELETE VARIABLES
        delete_var = ['id_fiscal', 'id_poliza', 'version_poliza', 'siniestro_descripcion', 'siniestro_factor_culpa',
                      'siniestro_hora_ocurrencia', 'siniestro_lugar', 'siniestro_cp', 'siniestro_poblacion',
                      'siniestro_provincia', 'siniestro_indicador_consorcio', 'siniestro_indicador_denuncia',
                      'siniestro_indicador_subrogacion', 'siniestro_indicador_via_judicial',
                      'siniestro_indicador_intervencion_policial', 'siniestro_indicador_posible_graciable',
                      'siniestro_indicador_concurrencia', 'siniestro_posible_fraude', 'siniestro_numero_perjudicados',
                      'siniestro_datos_testigos', 'siniestro_nombre_usuario_apertura',
                      'audit_siniestro_producto_tecnico',
                      'audit_siniestro_codigo_compania', 'siniestro_situacion']

        df = df.drop(*delete_var)

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.siniestro_output_prediction
        else:
            name = STRING.siniestro_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


if __name__ == '__main__':
    Siniestro(False).run()
