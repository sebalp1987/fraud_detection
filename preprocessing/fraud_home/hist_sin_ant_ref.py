import datetime
import time
import sys
import os

from pyspark.sql.functions import when, udf, lit, sum as sum_, datediff, count as count_
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, DateType, FloatType
from pyspark.sql.window import Window

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import outliers


class HistSinAntRef(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Historico Siniestro Anteriores Referencia")

    def run(self):
        df, df_base, bl_processed = self._extract_data()
        df = self._transform_data(df, df_base=df_base, bl_processed=bl_processed)
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
                    .csv(STRING.histsinantref_input_prediction, header=True, sep=',', nullValue='?'))

            df_base = (
                self._spark.read.csv(
                    STRING.histsinantref_input_training, header=True, sep=',', nullValue='?'
                ))
        else:

            df = (
                self._spark
                    .read
                    .csv(STRING.histsinantref_input_training, header=True, sep=',', nullValue='?'))

            df_base = None

        custom_schema = StructType([
            StructField("id_siniestro", IntegerType(), nullable=True),
            StructField("id_poliza", StringType(), nullable=True),
            StructField("fecha_apertura", IntegerType(), nullable=True),
            StructField("fecha_terminado", IntegerType(), nullable=True),
            StructField("nif_o_intm", StringType(), nullable=True),
            StructField("iban", StringType(), nullable=True),
            StructField("rol", StringType(), nullable=True),
            StructField("cod_rol", IntegerType(), nullable=True)
        ])

        file_list = [filename for filename in os.listdir(STRING.reporting_output) if filename.endswith('.csv')]
        bl_processed = (self._spark.
                        read.
                        csv(STRING.reporting_output + file_list[0], sep=';',
                            header=True,
                            encoding='UTF-8', schema=custom_schema))

        return df, df_base, bl_processed

    def _transform_data(self, df, df_base, bl_processed):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param bl_processed
        :return: Transformed DataFrame.
        """

        if self._is_diario:
            df = df.withColumn('TEST', lit(1))
            df_base = df_base.withColumn('TEST', lit(0))
            df = df.union(df_base)

        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro_ref')
        df = df.withColumn('id_siniestro_ref', df.id_siniestro_ref.cast(IntegerType()))
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))
        df = df.dropna(subset=['id_siniestro_ref'])
        df = df.dropna(subset=['id_siniestro'])

        # DATE VARIABLES FORMAT
        fecha_variables = ['hist_siniestro_otro_fecha_ocurrencia', 'hist_siniestro_fecha_terminado']
        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())
        for col in fecha_variables:
            df = df.fillna({col: '1900/01/01'})
            df = df.withColumn(col, func(df[col]))
            df = df.withColumn(col, when(df[col] == '1900-01-01', None).otherwise(df[col]))
            df = df.filter(df[col] <= time.strftime('%Y-%m-%d'))

        # COUNT ID_SINIESTRO_REF CUANTOS SINIESTROS TIENE
        df = df.withColumn('hist_sin_otros_count_version', lit(1))
        w = (Window().partitionBy(df.id_siniestro_ref).rowsBetween(-sys.maxsize, sys.maxsize))
        df = df.withColumn('hist_sin_otros_count', count_(df.hist_sin_otros_count_version).over(w))

        # SINIESTRO TERMINADO: We transform in dummy variables the category siniestro_sit
        types = df.select('hist_siniestro_otro_sit').distinct().collect()
        types = [ty['hist_siniestro_otro_sit'] for ty in types]
        type_list = [when(df['hist_siniestro_otro_sit'] == ty, 1).otherwise(0).alias('d_hist_siniestro_otro_sit_' + ty)
                     for ty in types]
        df = df.select(list(df.columns) + type_list)

        # DUMMIES ACUMULATIVAS
        types = ['d_hist_siniestro_otro_sit_' + x for x in types]
        var_dummies = ["hist_siniestro_otro_rehusado", "hist_siniestro_otro_bbdd",
                       "hist_siniestro_otro_unidad_investigacion", "hist_siniestro_otro_incidencia_tecnica",
                       "hist_siniestro_otro_incidencia_tecnica_positiva", "hist_siniestro_otro_incidencias",
                       "hist_siniestro_otro_cobertura", "hist_siniestro_otro_rehabilitado"] + types
        for col in var_dummies:
            df = df.withColumn(col + '_count', sum_(df[col]).over(w))
            df = df.drop(col)

        # DATE VARIABLES
        # Duración = Fecha_Terminado - Fecha_Ocurrrencia
        df = df.withColumn('hist_otros_fecha_apertura_terminado', datediff('hist_siniestro_fecha_terminado',
                                                                           'hist_siniestro_otro_fecha_ocurrencia'))

        df = df.withColumn('hist_otros_fecha_apertura_terminado',
                           sum_(df['hist_otros_fecha_apertura_terminado']).over(w))

        # Duración Promedio
        df = df.withColumn('hist_otros_duracion_promedio_sin', df['hist_otros_fecha_apertura_terminado'] /
                           df['hist_sin_otros_count'])

        # Último Siniestro de la póliza: We are going to keep the first row, it is the last sinister.
        df = df.withColumnRenamed('hist_siniestro_otro_fecha_ocurrencia', 'hist_siniestro_otro_ultimo_fecha_ocurrencia')
        df = df.orderBy('hist_siniestro_otro_ultimo_fecha_ocurrencia', ascending=False)

        # FUE UN SINIESTRO FRAUDULENTO? We check if the id_siniestro is associated with a previous Fraud Sinister
        bl_processed = bl_processed.select('id_siniestro').dropDuplicates(subset=['id_siniestro'])
        bl_processed = bl_processed.withColumn('hist_sin_otro_fraude_count', lit(1))
        df = df.join(bl_processed, on='id_siniestro', how='left')
        df = df.withColumn('hist_sin_otro_fraude_count', when(df['hist_sin_otro_fraude_count'].isNull(), 0).otherwise(
            df['hist_sin_otro_fraude_count']))
        df = df.withColumn('hist_sin_otro_fraude_count', sum_(df['hist_sin_otro_fraude_count']).over(w))

        # CARGA SINIESTRAL
        df = df.withColumnRenamed('coste_del_siniestro_por_rol', 'hist_siniestro_carga_siniestral')
        df = df.fillna({'hist_siniestro_carga_siniestral': 0})
        df = df.withColumn('hist_siniestro_carga_siniestral', df.hist_siniestro_carga_siniestral.cast(FloatType()))

        # Construimos el outlier a nivel siniestro: Luego hacemos la suma de los casos de outlier por id_siniestro_ref
        df = outliers.Outliers.outliers_mad(df, 'hist_siniestro_carga_siniestral', not_count_zero=True)
        df = df.withColumn('hist_siniestro_carga_siniestral_mad_outlier_count',
                           sum_(df['hist_siniestro_carga_siniestral_mad_outlier']).over(w))

        # suma total: Sumamos el total de la carga siniestral
        df = df.withColumn('hist_siniestro_carga_siniestral_sum', sum_(df['hist_siniestro_carga_siniestral']).over(w))

        # promedio
        df = df.withColumn('hist_sin_carga_siniestral_promedio', df['hist_siniestro_carga_siniestral_sum']
                           / df['hist_sin_otros_count'])

        # COBERTURAS
        # Outliers
        df = outliers.Outliers.outliers_mad(df, 'hist_siniestro_coberturas_involucradas', not_count_zero=True)
        df = df.withColumn('hist_siniestro_coberturas_involucradas_mad_outlier',
                           when(df['hist_siniestro_coberturas_involucradas'] > 3, 1).otherwise(0))
        df = df.withColumn('hist_siniestro_coberturas_involucradas_mad_outlier_count',
                           sum_(df['hist_siniestro_coberturas_involucradas_mad_outlier']).over(w))
        df = df.drop('hist_siniestro_coberturas_involucradas_mad_outlier')

        # promedio
        df = df.withColumn('hist_sin_otros_cober_sum', sum_(df['hist_siniestro_coberturas_involucradas']).over(w))
        df = df.withColumn('hist_sin_otros_cober_promedio', df['hist_sin_otros_cober_sum'] / df['hist_sin_otros_count'])

        # pagas-cubiertas
        df = df.withColumn('hist_siniestro_coberturas_involucradas_pagadas_sum',
                           sum_(df['hist_siniestro_coberturas_involucradas_pagadas']).over(w))
        df = df.withColumn('hist_sin_otros_pagas_cubiertas',
                           df['hist_siniestro_coberturas_involucradas_pagadas_sum'] / df['hist_sin_otros_cober_sum'])


        # no-pagas
        df = df.withColumn('hist_sin_otros_cob_no_pagas', df['hist_siniestro_coberturas_involucradas'] -
                           df['hist_siniestro_coberturas_involucradas_pagadas'])
        df = df.withColumn('hist_sin_otros_cob_no_pagas', sum_(df['hist_sin_otros_cob_no_pagas']).over(w))

        # VARIABLES DEL
        del_variables = ['id_fiscal', 'hist_siniestro_otro_descripcion', "hist_siniestro_duracion",
                         'hist_siniestro_fecha_terminado', 'hist_sin_otros_count_version',
                         'hist_siniestro_otro_oficina_productora',
                         'hist_siniestro_carga_siniestral', 'hist_siniestro_coberturas_involucradas',
                         'hist_siniestro_coberturas_involucradas_pagadas',
                         'hist_otros_fecha_apertura_terminado',
                         'auditFechaAperturaSiniestroReferencia', 'id_siniestro', 'id_poliza', 'version_poliza',
                         'audit_siniestro_producto_tecnico',
                         'audit_siniestro_codigo_compania', 'hist_siniestro_otro_sit'
                         ]

        df = df.drop(*del_variables)
        df = df.withColumnRenamed('id_siniestro_ref', 'id_siniestro')

        # Tomamos el primero de cada uno de los siniestros
        df = df.dropDuplicates(subset=['id_siniestro'])

        if self._is_diario:
            df = df.filter(df['TEST'] == 1)
            df = df.drop('TEST')

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.histsinantref_output_prediction
        else:
            name = STRING.histsinantref_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main para test
if __name__ == '__main__':
    HistSinAntRef(is_diario=True).run()
