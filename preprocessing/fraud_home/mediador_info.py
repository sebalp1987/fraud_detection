import datetime
import sys
import os

from pyspark.sql.functions import when, udf, regexp_replace, lit, sum as sum_, datediff, round as round_
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, DateType
from pyspark.sql.window import Window

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import functions as f, outliers


class MediadorInfo(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Mediador info")

    def run(self):
        df, bl_processed = self._extract_data()
        df = self._transform_data(df, bl_processed)
        self._load_data(df)

    def _extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """

        df = (
            self._spark
                .read
                .csv(STRING.mediador_input_training, header=True, sep=',', nullValue='?'))

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

        return df, bl_processed

    @staticmethod
    def _transform_data(df, bl_processed):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param bl_processed
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumn('mediador_cod_intermediario', df.mediador_cod_intermediario.cast(IntegerType()))
        df = df.orderBy('mediador_cod_intermediario')

        # Count Productos
        df = df.withColumn('pondera_producto', lit(1))
        w_cod_intermediario = (
            Window().partitionBy(df.mediador_cod_intermediario).rowsBetween(-sys.maxsize, sys.maxsize))
        df = df.withColumn('mediador_producto_count', sum_(df['pondera_producto']).over(w_cod_intermediario))

        # Blacklist
        bl_processed_mediador = bl_processed.filter(bl_processed['cod_rol'] == 3)
        bl_processed_mediador = bl_processed_mediador.withColumn('pondera', lit(1))
        bl_processed_mediador = bl_processed_mediador.select(['nif_o_intm', 'pondera'])
        w_mediador = (Window().partitionBy(bl_processed_mediador['nif_o_intm']).rowsBetween(-sys.maxsize, sys.maxsize))
        bl_processed_mediador = bl_processed_mediador.withColumn('mediador_cod_count_blacklist',
                                                                 sum_(bl_processed_mediador['pondera']).over(
                                                                     w_mediador))
        bl_processed_mediador = bl_processed_mediador.dropDuplicates(subset=['nif_o_intm'])
        df = df.join(bl_processed_mediador, df.mediador_cod_intermediario == bl_processed_mediador.nif_o_intm,
                     how='left')
        df = df.drop(*['nif_o_intm', 'pondera'])
        df = df.fillna({'mediador_cod_count_blacklist': 0})

        # Estado del mediador
        estado = {'1': 'Activo', '2': 'Activo', '3': 'Inactivo', '4': 'Pendiente', '5': 'Tramite'}
        funct = udf(lambda x: f.replace_dict(x, key_values=estado, key_in_value=True), StringType())
        df = df.withColumn('mediador_estado', funct(df['mediador_estado']))

        # Dummies var
        dummy_var = ['mediador_clase_intermediario', 'mediador_estado', 'id_agrup_producto']
        for col in dummy_var:
            df = df.fillna({col: 'No Identificado'})
            df = df.withColumn(col, regexp_replace(col, '0', ''))
            type_d = df.select(col).distinct().collect()
            type_d = [ty[col] for ty in type_d]
            col_list = [when(df[col] == ty, 1).otherwise(0).alias('d_' + col + '_' + ty) for ty in type_d]
            df = df.select(list(df.columns) + col_list)
            df.drop(col)

        # We sum the number of products
        product_names = [x for x in list(df.columns) if x.startswith('d_id_agrup_producto')]
        for col in product_names:
            df = df.withColumn(col, sum_(df[col]).over(w_cod_intermediario))

        # Antiguedad del mediador
        today = datetime.date.today()

        funct = udf(lambda x: datetime.datetime.strptime(x, '%Y%m%d'), DateType())
        df = df.withColumn('mediador_fecha_alta', when(df['mediador_fecha_alta'].isNull(), '19000101').otherwise(
            df['mediador_fecha_alta']))
        df = df.withColumn('mediador_fecha_alta', funct(df['mediador_fecha_alta']))
        df = df.withColumn('fecha_hoy', lit(today))
        df = df.withColumn('mediador_antiguedad', round_(datediff(df['fecha_hoy'], df['mediador_fecha_alta']) / 365))
        df = df.drop('fecha_hoy')

        # Accumulative float variables
        mediador_nan = df.filter(df['mediador_numero_polizas'].isNull())  # Guardamos los mediadores que tienen nulos
        df = df.dropna(subset=['mediador_numero_polizas'])
        float_variables = ['mediador_numero_polizas', 'mediador_numero_polizas_vigor', 'mediador_numero_siniestros',
                           'mediador_numero_siniestros_fraude',
                           'mediador_numero_siniestros_pagados']
        for col in float_variables:
            name_count = col + '_count'
            df = df.withColumn(name_count, sum_(df[col].cast(IntegerType())).over(w_cod_intermediario))

        # COUNT BY HOGAR: We now create an additional table to get the statistics for PROPERTY and then remarge
        # (0000000002;HOGAR)
        df_hogar = df.filter(df['id_agrup_producto'] == 2)
        df_hogar = df_hogar.select(*['mediador_cod_intermediario'] + float_variables)
        float_cols = [when(df_hogar[col].isNull(), 0).otherwise(df_hogar[col]).alias(col + '_hogar') for col in
                      float_variables]

        df_hogar = df_hogar.select(['mediador_cod_intermediario'] + float_cols)
        df = df.dropDuplicates(subset=['mediador_cod_intermediario'])
        df = df.join(df_hogar, on='mediador_cod_intermediario', how='left')

        # 1) STATISTICS SAME SET BY ROW: Here we compare the same set (GLOBAL or HOGAR) respecto to their
        # respectives columns
        poliza_var = ['mediador_numero_polizas', 'mediador_numero_polizas_vigor']
        siniestro_var = ['mediador_numero_siniestros',
                         'mediador_numero_siniestros_fraude',
                         'mediador_numero_siniestros_pagados']

        # a) Global: Need to use '_count'
        # polizas_vigor / polizas_total
        df = df.withColumn('mediador_poliza_vigor_total', df['mediador_numero_polizas_vigor_count'] /
                           df['mediador_numero_polizas_count'])

        # siniestros / siniestros total
        df = df.withColumn('mediador_siniestros_fraude_total', df['mediador_numero_siniestros_fraude_count'] /
                           df['mediador_numero_siniestros_count'])

        df = df.withColumn('mediador_siniestros_pagados_total', df['mediador_numero_siniestros_pagados_count'] /
                           df['mediador_numero_siniestros_count'])

        # siniestros / poliza
        for sin_str in siniestro_var:
            name = sin_str + '/poliza'
            df = df.withColumn(name, df[sin_str + '_count'] / df['mediador_numero_polizas_count'])

        # b) Hogar: Need to use '_hogar'
        # polizas_vigor / polizas_total
        df = df.withColumn('mediador_poliza_vigor_total_hogar', df['mediador_numero_polizas_vigor_hogar'] /
                           df['mediador_numero_polizas_hogar'])

        # siniestros / siniestros total

        df = df.withColumn('mediador_siniestros_fraude_total_hogar', df['mediador_numero_siniestros_fraude_hogar'] /
                           df['mediador_numero_siniestros_hogar'])

        df = df.withColumn('mediador_siniestros_pagados_total_hogar', df['mediador_numero_siniestros_pagados_hogar'] /
                           df['mediador_numero_siniestros_hogar'])

        # siniestros / poliza
        for sin_str in siniestro_var:
            name = sin_str + '/poliza_hogar'
            df = df.withColumn(name, df[sin_str + '_hogar'] / df['mediador_numero_polizas_hogar'])

        # 2) STATISTICS SAME SET BY COLUMN: We compare the relative weight to the specific column.
        # a) Global:
        for i in poliza_var:
            var = i + '_count'
            name = i + '_weight'
            # Solucion al error de array_list
            total = df.select(var).groupBy().agg(sum_(var).alias("total")).collect()
            df = df.withColumn(name, df[var] / total[0].total)

        for i in siniestro_var:
            var = i + '_count'
            name = i + '_weight'
            total = df.select(var).groupBy().agg(sum_(var).alias("total")).collect()
            df = df.withColumn(name, df[var] / total[0].total)

        # b) Hogar:
        for i in poliza_var:
            var = i + '_hogar'
            name = i + '_weight'
            total = df.select(var).groupBy().agg(sum_(var).alias("total")).collect()
            df = df.withColumn(name, df[var] / total[0].total)

        for i in siniestro_var:
            var = i + '_hogar'
            name = var + '_weight'
            total = df.select(var).groupBy().agg(sum_(var).alias("total")).collect()
            df = df.withColumn(name, df[var] / total[0].total)

        # STATISTICS DIFFERENT SETS BY COLUMN: Here we compare the relatives between HOGAR / GLOBAL
        for i in poliza_var:
            global_var = i + '_count'
            hogar_var = i + '_hogar'
            name = i + 'hogar/total'
            df = df.withColumn(name, df[hogar_var] / df[global_var])

        for i in siniestro_var:
            global_var = i + '_count'
            hogar_var = i + '_hogar'
            name = i + 'hogar/total'
            df = df.withColumn(name, df[hogar_var] / df[global_var])

        # OUTLIERS
        for i in poliza_var:
            global_var = i + '_count'
            hogar_var = i + '_hogar'
            df = outliers.Outliers.outliers_mad(df, global_var, not_count_zero=False)
            df = outliers.Outliers.outliers_mad(df, hogar_var, not_count_zero=False)

        for i in siniestro_var:
            global_var = i + '_count'
            hogar_var = i + '_hogar'
            df = outliers.Outliers.outliers_mad(df, global_var, not_count_zero=False)
            df = outliers.Outliers.outliers_mad(df, hogar_var, not_count_zero=False)

        # Unimos otra vez los mediadores que tienen nulos
        for col in list(df.columns):
            if col not in list(mediador_nan.columns):
                mediador_nan = mediador_nan.withColumn(col, lit(None))

        df = df.union(mediador_nan)

        # Outlier para el número de veces en la BL
        df = outliers.Outliers.outliers_mad(df, 'mediador_cod_count_blacklist', not_count_zero=True)

        # DELETE VARIABLES
        del_variables = float_variables + ['mediador_denominacion_intermediario', 'mediador_nif_intermediario',
                                           'mediador_fecha_alta', 'Description', 'pondera_producto']
        df = df.drop(*del_variables)
        df = df.fillna(0)

        return df

    @staticmethod
    def _load_data(df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(STRING.mediador_output_training)


# Main para test
if __name__ == '__main__':
    MediadorInfo(is_diario=True).run()
