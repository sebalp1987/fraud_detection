import os

from pyspark.sql.functions import when, udf
from pyspark.sql.types import IntegerType, StringType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import functions as f

class PolizaInfo(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session()

    def run(self):
        df, mediador = self._extract_data()
        df = self._transform_data(df, mediador)
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
                    .csv(STRING.poliza_input_prediction, header=True, sep=',', nullValue='?'))
        else:
            df = (
                self._spark
                    .read
                    .csv(STRING.poliza_input_training, header=True, sep=',', nullValue='?'))

        file_list = [filename for filename in os.listdir(STRING.mediador_output_training) if filename.endswith('.csv')]
        mediador = self._spark.read.csv(STRING.mediador_output_training+ file_list[0], header=True, sep=';')

        return df, mediador

    @staticmethod
    def _transform_data(df, mediador):
        """Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumnRenamed('audit_siniestro_referencia', 'id_siniestro')
        df = df.dropna(subset=['id_siniestro'])
        df = df.dropna(subset=['id_poliza'])

        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))
        mediador = mediador.withColumn('mediador_cod_intermediario',
                                       mediador.mediador_cod_intermediario.cast(IntegerType()))

        # MEDIADOR PRIMER RECIBO and SUCESIVOS do not match
        df = df.withColumn('poliza_mediador_primero_sucesivo_no_coincidencia',
                           when(df['poliza_id_mediador_gestor_primer_recibo'] != df[
                               'poliza_id_mediador_gestor_recibo_sucesivo'], 1).otherwise(0))

        # MATCH WITH MEDIADOR FILE
        df = df.join(mediador, df.poliza_cod_intermediario == mediador.mediador_cod_intermediario, how='left')
        for col in list(mediador.columns):
            df = df.fillna({col: -1})
        df = df.drop('mediador_cod_intermediario')

        # POLIZA CANAL: We match with the description

        df = df.withColumn('poliza_canal', df['poliza_canal'].cast(IntegerType()))
        canal = {1: 'Mediador', 5: 'Colectivo', 6: 'Grandes_Dist', 7: 'Deutsche', 9: 'Vida', 10: 'CAS'}
        function_dict = udf(lambda x: f.replace_dict_int(x, key_values=canal), StringType())
        df = df.withColumn('poliza_canal', function_dict(df['poliza_canal']))

        # POLIZA CREDIT SCORING: We match with the description
        df = df.fillna({'poliza_credit_scoring': 'No Informado'})

        # CATEGORICAL VAR: We transform categorical values to DUMMIES
        categorical_var = ['poliza_desc_estructura', 'poliza_canal', 'poliza_duracion', 'poliza_credit_scoring',
                           'poliza_ultimo_movimiento']
        for col in categorical_var:
            type_cols = df.select(col).distinct().collect()
            type_cols = [ty[col] for ty in type_cols]
            type_cols_list = [when(df[col] == ty, 1).otherwise(0).alias('d_' + col + '_' + ty) for ty in type_cols if
                              ty is not None]
            df = df.select(list(df.columns) + type_cols_list)
            df = df.drop(col)

        # CESION DE DERECHOS: Existe cesión de derechos?
        df = df.withColumn('poliza_cesion_derechos', when(df['poliza_cesion_derechos'] == 'No', 0).otherwise(1))

        # DELETE USELESS VAR
        delete_var = ['id_fiscal', 'id_poliza', 'id_producto', 'poliza_cod_comercial', 'poliza_entidad_legal',
                      'poliza_descripcion_producto', 'poliza_nif_intermediario',
                      'poliza_id_mediador_productor_secundario', 'poliza_mediador_denominacion_productor_secundario',
                      'poliza_nif_intermediario_secundario', 'poliza_id_mediador_productor_tercero',
                      'poliza_mediador_denominacion_productor_tercero', 'poliza_nif_intermediario_tercero',
                      'poliza_mediador_denominacion_gestor_primer_recibo', 'poliza_nif_intermediario_primer_recibo',
                      'poliza_mediador_denominacion_gestor_recibo_sucesivo',
                      'poliza_nif_intermediario_recibo_sucesivos',
                      'poliza_cod_estructura', 'poliza_codigo_negocio', 'poliza_nombre_negocio',
                      'poliza_cod_intermediario', 'poliza_denominacion_intermediario', 'mediador_cod_intermediario',
                      'mediador_cod_intermediario_sucesivos_recibo', 'poliza_motivo_ultimo_movimiento',
                      'poliza_id_mediador_gestor_primer_recibo', 'poliza_id_mediador_gestor_recibo_sucesivo',
                      'mediador_clase_intermediario', 'mediador_estado'

                      ]

        df = df.drop(*delete_var)

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.poliza_output_prediction
        else:
            name = STRING.poliza_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


if __name__ == '__main__':
    PolizaInfo(True).run()
