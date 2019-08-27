import datetime
import time
import sys
import os

from pyspark.sql.functions import when, udf, lit, sum as sum_, datediff, count as count_, col
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, DateType, FloatType
from pyspark.sql.window import Window

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import outliers


class HistSinAntOtras(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Historico Siniestro Anteriores Otras Pólizas")

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
                    .csv(STRING.histsinantotras_input_prediction, header=True, sep=',', nullValue='?'))

            df_base = (
                self._spark.read.csv(
                    STRING.histsinantotras_input_training, header=True, sep=',', nullValue='?'
                ))
        else:

            df = (
                self._spark
                    .read
                    .csv(STRING.histsinantotras_input_training, header=True, sep=',', nullValue='?'))

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
        exprs = [df[column].alias(column.replace('"', '')) for column in df.columns]
        df = df.select(*exprs)
        exprs = [df[column].alias(column.replace(' ', '')) for column in df.columns]
        df = df.select(*exprs)

        df = df.withColumnRenamed('hist_siniestro_poliza_otro_id_siniestro', 'id_siniestro')
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro_ref')
        df = df.withColumn('id_siniestro_ref', df.id_siniestro_ref.cast(IntegerType()))
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))
        df = df.dropna(subset=['id_siniestro_ref'])
        df = df.dropna(subset=['id_siniestro'])

        # DATE VARIABLES FORMAT
        fecha_variables = ["hist_siniestro_poliza_otro_fecha_ocurrencia", "hist_siniestro_poliza_otro_fecha_terminado",
                           "auditFechaAperturaSiniestroReferencia"]
        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())
        for col in fecha_variables:
            df = df.fillna({col: '1900/01/01'})
            df = df.withColumn(col, func(df[col]))
            df = df.withColumn(col, when(df[col] == '1900-01-01', None).otherwise(df[col]))
            df = df.filter(df[col] <= time.strftime('%Y-%m-%d'))

        # We check that the sinister in the other policy is before the reference sinister, because we want to know the
        # past values
        df = df.filter(df['auditFechaAperturaSiniestroReferencia'] >=
                       df['hist_siniestro_poliza_otro_fecha_ocurrencia'])

        # COUNT POLIZA-VERSION: We count how many sinisters before have the costumer. It counts how many times appear a
        # row in the table, because each line is referred to a unique sinister
        df = df.withColumn('hist_sin_poliza_otro_count_version', lit(1))
        w = (Window().partitionBy(df.id_siniestro_ref).rowsBetween(-sys.maxsize, sys.maxsize))
        df = df.withColumn('hist_sin_poliza_otro_count', count_(df.hist_sin_poliza_otro_count_version).over(w))

        # COUNT POLIZAS: We count how many policies has the customer. We have to construct another table so we can
        # group at the level of policies.
        count_poliza = df.select(['id_siniestro_ref', 'hist_siniestro_poliza_otro_id_poliza'])
        count_poliza = count_poliza.dropDuplicates()
        count_poliza = count_poliza.withColumnRenamed('hist_siniestro_poliza_otro_id_poliza',
                                                      'hist_sin_poliza_otro_count_polizas')
        count_poliza = count_poliza.withColumn('hist_sin_poliza_otro_count_polizas',
                                               count_(df['id_siniestro_ref']).over(w))
        count_poliza = count_poliza.dropDuplicates(subset=['id_siniestro_ref'])
        df = df.join(count_poliza, on='id_siniestro_ref', how='left')

        # SINIESTROS/POLIZAS: Here we calculate the ratio nºsinisters/nº policies
        df = df.withColumn('hist_siniestro_poliza_otro_siniestros_polizas',
                           df['hist_sin_poliza_otro_count'] / df['hist_sin_poliza_otro_count_polizas'])

        # FUE UN SINIESTRO FRAUDULENTO? We check if the id_siniestro is associated with a previous Fraud Sinister
        bl_processed = bl_processed.select('id_siniestro').dropDuplicates(subset=['id_siniestro'])
        bl_processed = bl_processed.withColumn('hist_sin_poliza_otro_fraude', lit(1))
        df = df.join(bl_processed, on='id_siniestro', how='left')
        df = df.withColumn('hist_sin_poliza_otro_fraude', when(df['hist_sin_poliza_otro_fraude'].isNull(), 0).otherwise(
            df['hist_sin_poliza_otro_fraude']))

        # POR PRODUCTO: We group the product number by predefined categories in tabla_productos. It permits a better
        # classification. Here we have to pre-process the product label format to have coincidence.
        types = df.select('hist_siniestro_poliza_otro_id_producto').distinct().collect()
        types = [ty['hist_siniestro_poliza_otro_id_producto'] for ty in types]
        types_list = [when(df['hist_siniestro_poliza_otro_id_producto'] == ty, 1).otherwise(0).alias(
            'd_hist_sin_poliza_otro_producto_' + ty) for ty in types]
        df = df.select(list(df.columns) + types_list)
        df.drop('hist_siniestro_poliza_otro_id_producto')

        # DUMMIES: We acumulate the dummy variables to get the variables at cod_filiacion level
        types = ['d_hist_sin_poliza_otro_producto_' + x for x in types]
        var_dummies = ["hist_siniestro_poliza_otro_bbdd", "hist_siniestro_poliza_otro_unidad_investigacion",
                       "hist_siniestro_poliza_otro_incidencia_tecnica",
                       "hist_siniestro_poliza_otro_incidencia_tecnica_positiva",
                       "hist_siniestro_poliza_otro_incidencias",
                       "hist_siniestro_poliza_otro_cobertura"] + types
        for col in var_dummies:
            df = df.withColumn(col + '_count', sum_(df[col]).over(w))
            df = df.drop(col)

        # FECHAS: We have two dates. fecha_ocurrencia and fecha_terminado. We have to take into account claims
        # that are not finished. If the claim is notfinished we input today as date
        # and create a variable that indicates the situation.
        df = df.withColumn('hist_siniestro_poliza_otro_no_terminado',
                           when(df['hist_siniestro_poliza_otro_fecha_terminado'].isNull(), 1).otherwise(0))
        df = df.fillna({'hist_siniestro_poliza_otro_fecha_terminado': time.strftime('%Y-%m-%d')})

        # Claim duration: We calculate the cumulated duration and the average duration.
        df = df.withColumn('hist_poliza_otro_fecha_apertura_terminado',
                           datediff('hist_siniestro_poliza_otro_fecha_terminado',
                                    'hist_siniestro_poliza_otro_fecha_ocurrencia'))
        df = df.withColumn('hist_poliza_otro_fecha_apertura_terminado',
                           sum_(df['hist_poliza_otro_fecha_apertura_terminado']).over(w))
        df = df.withColumn('hist_poliza_otro_duracion_promedio_sin',
                           df['hist_poliza_otro_fecha_apertura_terminado'] / df['hist_sin_poliza_otro_count'])

        # ULTIMO SINIESTRO DE LA POLIZA
        df = df.withColumnRenamed('hist_siniestro_poliza_otro_fecha_ocurrencia',
                                  'hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia')
        df = df.orderBy('hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia', ascending=False)

        # CARGA SINIESTRAL
        # Outlier: First we calculate the outliers quantity by cliente-sinister so we can get the intra-effect
        df = df.withColumnRenamed('coste_del_siniestro_por_rol', 'hist_siniestro_poliza_otro_carga_siniestral')
        df = df.fillna({'hist_siniestro_poliza_otro_carga_siniestral': 0})
        df = df.withColumn('hist_siniestro_poliza_otro_carga_siniestral',
                           df.hist_siniestro_poliza_otro_carga_siniestral.cast(FloatType()))

        # Construimos el outlier a nivel siniestro: Luego hacemos la suma de los casos de outlier por id_siniestro_ref
        df = outliers.Outliers.outliers_mad(df, 'hist_siniestro_poliza_otro_carga_siniestral', not_count_zero=True)
        df = df.withColumn('hist_siniestro_poliza_otro_carga_siniestral_mad_outlier_count',
                           sum_(df['hist_siniestro_poliza_otro_carga_siniestral_mad_outlier']).over(w))
        df = df.withColumn('hist_siniestro_poliza_otro_carga_siniestral_mad_outlier_promedio',
                           df['hist_siniestro_poliza_otro_carga_siniestral_mad_outlier_count'] / df[
                               'hist_sin_poliza_otro_count'])
        df = df.drop('hist_siniestro_poliza_otro_carga_siniestral_mad_outlier')

        # We calculate the sum and the average by sinister
        df = df.withColumn('hist_siniestro_poliza_otro_carga_siniestral_count',
                           sum_(df['hist_siniestro_poliza_otro_carga_siniestral']).over(w))
        df = df.withColumn('hist_siniestro_poliza_otro_carga_siniestral_promedio',
                           df['hist_siniestro_poliza_otro_carga_siniestral_count'] / df['hist_sin_poliza_otro_count'])

        # COBERTURAS
        # mayor a 3: we consider as outlier > 3, because the mean is concentrated around 1.28
        df = df.withColumn('hist_sin_poliza_otro_mayor3coberturas',
                           when(df["hist_siniestro_poliza_otro_coberturas_involucradas"] > 3, 1).otherwise(0))
        df = df.withColumn('hist_sin_poliza_otro_mayor3coberturas',
                           sum_(df['hist_sin_poliza_otro_mayor3coberturas']).over(w))

        # promedio: Average by claim
        df = df.withColumn('hist_sin_poliza_otro_cober_sum',
                           sum_(df['hist_siniestro_poliza_otro_coberturas_involucradas']).over(w))
        df = df.withColumn('hist_sin_poliza_otro_cober_promedio',
                           df["hist_sin_poliza_otro_cober_sum"] / df['hist_sin_poliza_otro_count'])

        # pagas-cubiertas: We calculate this at the coustomer cumulated level and not to claim level
        df = df.withColumn('hist_siniestro_poliza_otro_coberturas_involucradas_pagadas_sum',
                           sum_(df['hist_siniestro_poliza_otro_coberturas_involucradas_pagadas']).over(w))
        df = df.withColumn('hist_sin_poliza_otro_pagas_cubiertas',
                           df["hist_siniestro_poliza_otro_coberturas_involucradas_pagadas_sum"] / df[
                               'hist_sin_poliza_otro_cober_sum'])

        # no pagas: Here we calculate at the claim level, counting the total unpaid coverages
        df = df.withColumn('hist_sin_poliza_otro_cob_no_pagas',
                           when(df['hist_siniestro_poliza_otro_coberturas_involucradas_pagadas'] == 0, 1).otherwise(0))
        df = df.withColumn('hist_sin_poliza_otro_cob_no_pagas', sum_(df['hist_sin_poliza_otro_cob_no_pagas']).over(w))

        # DELETE VARIABLES: We delete variables that are not relevant or have been transformed
        del_variables = ['hist_siniestro_poliza_otro_id_poliza', 'hist_siniestro_poliza_otro_id_producto',
                         'hist_siniestro_poliza_otro_version', 'hist_siniestro_poliza_otro_id_siniestro',
                         'hist_siniestro_poliza_otro_fecha_terminado', 'hist_siniestro_poliza_otro_bbdd',
                         'hist_siniestro_poliza_otro_unidad_investigacion',
                         'hist_siniestro_poliza_otro_incidencia_tecnica',
                         'hist_siniestro_poliza_otro_incidencia_tecnica_positiva',
                         'hist_siniestro_poliza_otro_incidencias',
                         'hist_siniestro_poliza_otro_cobertura', 'hist_siniestro_poliza_otro_carga_siniestral',
                         'hist_siniestro_poliza_otro_coberturas_involucradas',
                         'hist_siniestro_poliza_otro_coberturas_involucradas_pagadas',
                         'id_fiscal', 'hist_sin_poliza_otro_count_version',
                         'Agrupación productos', 'Producto', 'auditFechaAperturaSiniestroReferencia',
                         'cliente_codfiliacion', 'audit_siniestro_codigo_compania', 'id_siniestro'
                         ]

        df = df.drop(*del_variables)
        df = df.withColumnRenamed('id_siniestro_ref', 'id_siniestro')
        df = df.dropDuplicates(subset=['id_siniestro'])

        # OUTLIER: We calculate the outliers referred to the ratio claims/policies.
        df = outliers.Outliers.outliers_mad(df, 'hist_siniestro_poliza_otro_siniestros_polizas', not_count_zero=False)

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
            name = STRING.histsinantotras_output_prediction
        else:
            name = STRING.histsinantotras_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main para test
if __name__ == '__main__':
    HistSinAntOtras(is_diario=False).run()
