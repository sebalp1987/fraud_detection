import sys
import datetime
from dateutil.relativedelta import relativedelta

from pyspark.sql.functions import when, udf, lit, sum as sum_, datediff
from pyspark.sql.types import IntegerType, StringType, DateType, FloatType
from pyspark.sql.window import Window

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import functions as f, outliers


class HistMovPolOtras(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Historico Movimiento Poliza Otras")

    def run(self):


        print(self._spark.sparkContext._conf.getAll())
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
                    .csv(STRING.histmovpolotras_input_prediction, header=True, sep=',', nullValue='?'))

            df_base = (
                    self._spark
                        .read
                        .csv(STRING.histmovpolotras_input_training, header=True, sep=',', nullValue='?'))
        else:
            df = (
                self._spark
                    .read
                    .csv(STRING.histmovpolotras_input_training, header=True, sep=',', nullValue='?'))

            df_base = None

        return df, df_base

    def _transform_data(self, df, df_base):
        """Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        if self._is_diario:
            df = df.withColumn('TEST', lit(1))
            df_base = df_base.withColumn('TEST', lit(0))
            df = df.union(df_base)
            del df_base

        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df['id_siniestro'].cast(IntegerType()))
        df = df.dropna(subset=['id_siniestro'])
        df = df.withColumnRenamed('auditFechaAperturaSiniestroReferencia', 'fecha_apertura_siniestro')

        # VARIABLES DE FECHA: We convert every date variable to a same format and we control the date
        variable_fecha = ["hist_poliza_fecha_inicio", "hist_poliza_fecha_efecto_natural",
                          "hist_poliza_fecha_efecto_mvto", "hist_poliza_fecha_vto_mvto",
                          "hist_poliza_vto_natural"
                          ]

        for col in variable_fecha:
            df = df.withColumn(col, when(df[col] == '1900-01-01', None).otherwise(df[col]))

        control_fecha = ["hist_poliza_fecha_inicio", "hist_poliza_fecha_efecto_natural"]
        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())
        for col in control_fecha:
            df = df.dropna(subset=[col])
            df = df.filter(df[col] <= STRING.DAY)
            df = df.filter(df[col] > (datetime.datetime.today() + relativedelta(years=-5)).strftime('%Y-%m-%d'))
        for col in variable_fecha:
            df = df.withColumn(col, func(df[col]))

        df = df.orderBy(['id_siniestro'], ascending=[True])

        # COUNT POLIZAS POR SINIESTRO: We weight the policies by sinister
        df = df.withColumn('pondera_poliza', lit(1))
        w = (Window().partitionBy(df.id_siniestro).rowsBetween(-sys.maxsize, sys.maxsize))
        df = df.withColumn('hist_mov_poliza_otro_count', sum_(df['pondera_poliza']).over(w))

        # POR PRODUCTO: We use an extra table to group the products
        types = df.select('hist_id_producto').distinct().collect()
        types = [ty['hist_id_producto'] for ty in types]
        types_list = [when(df['hist_id_producto'] == ty, 1).otherwise(0).alias(
            'd_d_hist_mov_poliza_otro_producto_' + ty) for ty in types]
        df = df.select(list(df.columns) + types_list)
        df.drop('hist_id_producto')

        # PROMEDIO DE VERSIONES: We count the versions by policy and then we get the average by policy.
        df = df.withColumn('hist_mov_poliza_otro_version_count',
                           sum_(df['hist_poliza_version'].cast(IntegerType())).over(w))
        df = df.withColumn('hist_mov_poliza_otro_version_promedioxpoliza', df['hist_mov_poliza_otro_version_count']
                           / df['hist_mov_poliza_otro_count'])

        # PROMEDIO DE SUPLEMENTOS: We count the suplements by policy and then we get the average by policy.
        df = df.withColumn('hist_mov_poliza_otro_suplemento_count',
                           sum_(df['hist_poliza_suplementos'].cast(IntegerType())).over(w))
        df = df.withColumn('hist_mov_poliza_otro_suplemento_promedioxpoliza',
                           df['hist_mov_poliza_otro_suplemento_count']
                           / df['hist_mov_poliza_otro_count'])

        # ULTIMO MOVIMIENTO: We group the policy last movement by ANULACION, CARTERA, REGULARIZACION or SUPLMENTO
        dict_replace = {'ANULACION': 'ANULACION', 'CARTERA': 'CARTERA', 'REGULARIZACION': 'REGULARIZACION',
                        'SUPLEMENTO': 'SUPLEMENTO'}
        func = udf(lambda x: f.replace_dict_starswith(x, key_values=dict_replace), StringType())
        df = df.withColumn('hist_poliz_ultimo_movimiento', func(df['hist_poliz_ultimo_movimiento']))

        # VARIABLES CATEGÓRICAS: We get dummies from categorical variables
        variable_categorica = ["hist_poliz_ultimo_movimiento",
                               "hist_poliza_estado"
                               ]
        variable_dummy = []
        for col in variable_categorica:
            types = df.select(col).distinct().collect()
            types = [ty[col] for ty in types]
            types_list = [when(df[col] == ty, 1).otherwise(0).alias('d_' + col + '_' + ty) for ty in types if
                          ty is not None]
            variable_dummy += ['d_' + col + '_' + ty for ty in types if ty is not None]
            df = df.select(list(df.columns) + types_list)

        variable_dummy += ['hist_poliza_sospechoso']
        for col in variable_dummy:
            name = col + '_count'
            df = df.fillna({col: 0})
            df = df.withColumn(name, sum_(df[col].cast(IntegerType())).over(w))

        # FECHAS
        # 1) Inicio-Efecto:
        # Efecto Natural de la Póliza - Fecha Inicio Póliza
        df = df.withColumn('hist_mov_poliza_otro_dif_inicio_efecto',
                           datediff('hist_poliza_fecha_efecto_natural', 'hist_poliza_fecha_inicio'))
        df = df.withColumn('hist_mov_poliza_otro_dif_inicio_efecto_negativo',
                           when(df['hist_mov_poliza_otro_dif_inicio_efecto'] < 0, 1).otherwise(0))
        df = df.withColumn('hist_mov_poliza_otro_dif_inicio_efecto_negativo',
                           sum_(df['hist_mov_poliza_otro_dif_inicio_efecto_negativo']).over(w))

        # Acumulamos la diferencia efecto-inicio, sacamos el promedio por siniestro de la póliza
        df = df.withColumn('hist_mov_poliza_otro_dif_inicio_efecto_sum',
                           sum_(df['hist_mov_poliza_otro_dif_inicio_efecto']).over(w))
        df = df.withColumn('hist_mov_poliza_otro_dif_inicio_efecto_promedio',
                           df['hist_mov_poliza_otro_dif_inicio_efecto_sum'] / df['hist_mov_poliza_otro_count'])

        # 2) Efecto- Efecto vto.
        # Vencimiento Natural - Efecto Natural de la Póliza
        df = df.withColumn('hist_mov_poliza_otro_dif_efecto_vto',
                           datediff('hist_poliza_vto_natural', 'hist_poliza_fecha_efecto_natural'))

        # Acumulamos vto-efecto y obtenemos el promedio por siniestro de la póliza
        df = df.withColumn('hist_mov_poliza_otro_dif_efecto_vto_promedio',
                           sum_(df['hist_mov_poliza_otro_dif_efecto_vto']).over(w))
        df = df.withColumn('hist_mov_poliza_otro_dif_efecto_vto_promedio',
                           df['hist_mov_poliza_otro_dif_efecto_vto_promedio'] / df['hist_mov_poliza_otro_count'])

        # VARIABLES INT: For the INT variables we acumulate and obtain the average by sinister.
        variables_int = ["hist_poliza_numero_siniestros"]
        for col in variables_int:
            count = col + '_count'
            promedio = col + '_promedio'
            df = df.withColumn(count, sum_(df[col].cast(IntegerType())).over(w))
            df = df.withColumn(promedio, df[count] / df['hist_mov_poliza_otro_count'])

        # CARGA SINIESTRAL
        df = df.withColumnRenamed('coste_del_siniestro_por_rol', 'hist_poliza_carga_siniestral')
        df = df.fillna({'hist_poliza_carga_siniestral': 0})
        df = df.withColumn('hist_poliza_carga_siniestral', df['hist_poliza_carga_siniestral'].cast(FloatType()))

        # VARIABLES FLOAT:
        variables_float = ["hist_poliza_carga_siniestral", "hist_poliza_recibos_pagos_importe"]
        # First, before group we calculate outliers. We do it before because we want to count the outliers sinister
        # by sinister. If we first group, we lost the intra-effect of the sinister
        for col in variables_float:
            df = df.withColumn(col, df[col].cast(FloatType()))
            df = outliers.Outliers.outliers_mad(df, col, not_count_zero=True)
            count = col + '_count'
            promedio = col + '_promedio'
            name_outlier = str(col) + '_mad_outlier'
            count_outlier = name_outlier + '_count'
            promedio_outlier = name_outlier + '_promedio'
            df = df.withColumn(count_outlier, sum_(df[name_outlier]).over(w))
            df = df.withColumn(promedio_outlier, df[count_outlier] / df['hist_mov_poliza_otro_count'])
            df = df.drop(name_outlier)
            df = df.withColumn(count, sum_(df[col]).over(w))
            df = df.withColumn(promedio, df[count] / df['hist_mov_poliza_otro_count'])

        df = df.dropDuplicates(subset=['id_siniestro'])

        # VARIABLES DEL
        variable_del = ['id_fiscal', 'id_poliza', 'hist_id_producto', 'hist_poliza_version',
                        'hist_poliza_suplementos', 'hist_poliz_ultimo_movimiento',
                        'hist_poliz_motivo_ultimo_movimiento',
                        'hist_poliza_estado', 'hist_poliza_fecha_inicio', 'hist_poliza_fecha_efecto_natural',
                        'hist_poliza_fecha_efecto_mvto', 'hist_poliza_fecha_vto_mvto', 'hist_poliza_vto_natural',
                        'hist_poliza_numero_siniestros', 'hist_poliza_carga_siniestral',
                        'hist_poliza_recibos_pagos_importe', 'fecha_apertura_siniestro', 'pondera_poliza',
                        'Agrupación productos',
                        'Producto', 'hist_mov_poliza_otro_dif_inicio_efecto', 'hist_mov_poliza_otro_dif_efecto_vto',
                        'cliente_codfiliacion'
                        ]

        df = df.drop(*variable_del)

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
            name = STRING.histmovpolotras_output_prediction
        else:
            name = STRING.histmovpolotras_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main para test
if __name__ == '__main__':
    HistMovPolOtras(is_diario=True).run()
