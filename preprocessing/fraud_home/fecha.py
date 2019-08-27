import datetime
import sys

from pyspark.sql import Row
from pyspark.sql.functions import when, udf, coalesce, year, month, dayofmonth, \
    date_format, lit, sum as sum_, count as count_, datediff
from pyspark.sql.types import IntegerType, DateType
from pyspark.sql.window import Window

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import checklist_spark, STRING


class Fecha(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Fecha")

    def run(self):
        # self.logger.info("Running Fecha")
        print("Running Fecha")
        df, df_reserva, df_reserva_new, df_fecha = self.extract_data()
        df = self.transform_data(df, df_reserva=df_reserva, df_reserva_new=df_reserva_new, df_fecha=df_fecha,
                                 init_date_new_="2017-01-01", init_date_historic_="2014-01-01")
        self.load_data(df)
        self._spark.stop()

    def extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """
        if self._is_diario:
            df = (
                self._spark
                    .read
                    .csv(STRING.fecha_input_prediction, header=True, sep=',', nullValue='?'))

            df_reserva = (
                self._spark
                    .read
                    .csv(STRING.poreservable_input_training, header=True, sep=',', nullValue='?'))

            df_reserva_new = (
                self._spark
                    .read
                    .csv(STRING.poreservable_input_prediction, header=True, sep=',', nullValue='?'))

            df_fecha = (
                self._spark
                    .read
                    .csv(STRING.fecha_input_training, header=True, sep=',',
                         ))

        else:
            df = (
                self._spark
                    .read
                    .csv(STRING.fecha_input_training, header=True, sep=',', nullValue='?'))

            df_reserva = (
                self._spark
                    .read
                    .csv(STRING.poreservable_input_training, header=True, sep=',', nullValue='?'))

            df_reserva_new = None

            df_fecha = None

        return df, df_reserva, df_reserva_new, df_fecha

    def transform_data(self, df, df_reserva, df_reserva_new, df_fecha, init_date_new_, init_date_historic_):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param df_reserva
        :param df_reserva_new
        :param df_fecha
        :param init_date_new_: Minimun date for new claims
        :param init_date_historic_: Max historical data
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # CONSERVED VARIABLES: We drop the variables that are not well defined or that at wrong defined.
        var_conserved = ["id_siniestro", 'id_poliza', 'version_poliza', "fecha_poliza_emision",
                         "fecha_poliza_efecto_natural", "fecha_poliza_efecto_mvto", "fecha_poliza_vto_movimiento",
                         "fecha_poliza_vto_natural", "fecha_siniestro_ocurrencia", 'fecha_siniestro_comunicacion',
                         "fecha_primera_visita_peritaje",
                         "fecha_ultima_visita_peritaje"]

        df = df.select(*var_conserved)

        # We fill siniestro_comunicacion with siniestro_ocurrencia
        df = df.withColumn('fecha_siniestro_comunicacion',
                           coalesce('fecha_siniestro_comunicacion', 'fecha_siniestro_ocurrencia'))

        # STRIP dates: YEAR, MONTH, WEEKDAY, DAY
        var_fecha = ["fecha_poliza_emision", "fecha_poliza_efecto_natural",
                     "fecha_poliza_efecto_mvto", "fecha_poliza_vto_movimiento",
                     "fecha_poliza_vto_natural",
                     "fecha_siniestro_ocurrencia",
                     'fecha_primera_visita_peritaje',
                     'fecha_ultima_visita_peritaje', 'fecha_siniestro_comunicacion'
                     ]

        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())

        for col in var_fecha:
            year_name = str(col) + '_year'
            month_name = str(col) + '_month'
            day_name = str(col) + '_day'
            weekday_name = str(col) + '_weekday'
            df = df.fillna({col: '1900/01/01'})
            df = df.withColumn(col, func(df[col]))
            df = df.withColumn(col, when(df[col] == '1900-01-01', None).otherwise(df[col]))
            df = df.withColumn(year_name, year(df[col]))
            df = df.withColumn(month_name, month(df[col]))
            df = df.withColumn(day_name, dayofmonth(df[col]))
            df = df.withColumn(weekday_name, date_format(col, 'u') - 1)  # We adapt to (0=Monday, 1=Tuesday...)
            df = df.withColumn(weekday_name, df[weekday_name].cast(IntegerType()))

        # Filtering by INIT_DATE parameter
        df = df.filter(df['fecha_siniestro_ocurrencia'] >= init_date_historic_)

        # CHECKLIST 6a
        df = df.withColumn('checklist6a', lit(0))
        df = df.withColumn('checklist6a_PP', lit(0))

        # CHECKLIST 6b
        if self._is_diario:
            # Filtering new Claims INIT_DATE
            df = df.filter(df['fecha_siniestro_comunicacion'] >= init_date_new_)
            auxiliar_list = checklist_spark.checklist6b(df, df_fecha, df_reserva_new, df_reserva)

        else:
            auxiliar_list = checklist_spark.checklist6b(None, df, None, df_reserva)

        if auxiliar_list:
            r = Row('id_siniestro_c', 'checklist_6b')
            df_claims = self._spark.createDataFrame(r(i, x) for i, x in auxiliar_list)
            df = df.join(df_claims, df.id_siniestro == df_claims.id_siniestro_c, how='left')
            del df_claims, r, auxiliar_list

            df = df.drop('id_siniestro_c')
            df = df.fillna({'checklist_6b': 0})
        else:
            df = df.withColumn('checklist_6b', lit(0))

        # CHECKLIST 7
        if self._is_diario:
            auxiliar_list = checklist_spark.checklist_7(df, df_fecha, df_reserva_new, df_reserva)
        else:
            auxiliar_list = checklist_spark.checklist_7(None, df, None, df_reserva)

        if auxiliar_list:
            r = Row('id_siniestro', 'checklist_7')
            df_claims = self._spark.createDataFrame(r(i, x) for i, x in auxiliar_list)
            del auxiliar_list, r

            df = df.join(df_claims, on='id_siniestro', how='left')
            del df_claims
            df = df.drop('id_siniestro_c')
            df = df.fillna({'checklist_7': 0})
        else:
            df = df.withColumn('checklist_7', lit(0))

        # CHECKLIST 14
        if self._is_diario:
            auxiliar_list = checklist_spark.checklist_14(df, df_fecha, df_reserva_new, df_reserva)
        else:
            auxiliar_list = checklist_spark.checklist_14(None, df, None, df_reserva)

        if auxiliar_list:
            r = Row('id_siniestro_c', 'checklist_14')
            df_claims = self._spark.createDataFrame(r(i, x) for i, x in auxiliar_list)

            w = (Window().partitionBy(df_claims.id_siniestro_c).rowsBetween(-sys.maxsize, sys.maxsize))
            df_claims = df_claims.withColumn('checklist_14_coberturas_repetidas', sum_(df_claims.checklist_14).over(w))
            df_claims = df_claims.withColumn('checklist_14_siniestros_involucrados',
                                             count_(df_claims.checklist_14).over(w))
            df_claims = df_claims.dropDuplicates(subset=['id_siniestro_c'])
            df_claims.drop('checklist_14')
            df = df.join(df_claims, df.id_siniestro == df_claims.id_siniestro_c, how='left')
            del df_claims, r, auxiliar_list
            df = df.drop('id_siniestro_c')
            df = df.fillna({'checklist_14_coberturas_repetidas': 0})
            df = df.fillna({'checklist_14_siniestros_involucrados': 0})
        else:
            df = df.withColumn('checklist_14_coberturas_repetidas', lit(0))
            df = df.withColumn('checklist_14_siniestros_involucrados', lit(0))

        # COMPLEX NON-COMPLEX VARIABLES: We define two types of dates. That dates we want more detail we generate
        # every type of possible variable. Non-complex will be more agroupated variables.
        var_fecha_complex = ["fecha_siniestro_ocurrencia"]
        var_fecha_less_complex = ["fecha_poliza_efecto_natural",
                                  "fecha_poliza_vto_natural"]

        for i in var_fecha_complex:
            # We create dummies
            col_names = [str(i) + '_year', str(i) + '_month', str(i) + '_weekday']
            for col in col_names:
                types = df.select(col).distinct().collect()
                types = [ty[col] for ty in types]
                type_list = [when(df[col] == ty, 1).otherwise(0).alias('d_' + col + '_' + str(ty)) for ty in types]
                df = df.select(list(df.columns) + type_list)

            # days range
            day = str(i) + '_day'
            df = df.withColumn(day + '1_10', when(df[day].between(1, 10), 1).otherwise(0))
            df = df.withColumn(day + '10_20', when(df[day].between(11, 20), 1).otherwise(0))
            df = df.withColumn(day + '20_30', when(df[day].between(21, 31), 1).otherwise(0))

        for i in var_fecha_less_complex:
            # month in holiday
            df = df.withColumn(str(i) + '_month_holiday', when(df[str(i) + '_month'].isin([1, 8, 12]), 1).otherwise(0))

            # days range
            day = str(i) + '_day'
            df = df.withColumn(day + '1_10', when(df[day].between(1, 10), 1).otherwise(0))
            df = df.withColumn(day + '10_20', when(df[day].between(11, 20), 1).otherwise(0))
            df = df.withColumn(day + '20_30', when(df[day].between(21, 31), 1).otherwise(0))

            # weekend or monday
            df = df.withColumn(str(i) + '_weekday_weekend', when(df[str(i) + '_weekday'].isin([6, 7]), 1).otherwise(0))
            df = df.withColumn(str(i) + '_weekday_monday', when(df[str(i) + '_weekday'] == 0, 1).otherwise(0))

            # FIRST DELETE: We delete that variables we generated before that are not relevant or are
            # too specific.

        del_variables = ['fecha_poliza_emision_year',
                         'fecha_poliza_emision_month',
                         'fecha_poliza_emision_day',
                         'fecha_poliza_emision_weekday',
                         'fecha_poliza_efecto_natural_year',
                         'fecha_poliza_efecto_natural_month',
                         'fecha_poliza_efecto_natural_day',
                         'fecha_poliza_efecto_natural_weekday',
                         'fecha_poliza_efecto_mvto_year',
                         'fecha_poliza_efecto_mvto_month',
                         'fecha_poliza_efecto_mvto_day',
                         'fecha_poliza_efecto_mvto_weekday',
                         'fecha_poliza_vto_movimiento_year',
                         'fecha_poliza_vto_movimiento_month',
                         'fecha_poliza_vto_movimiento_day',
                         'fecha_poliza_vto_movimiento_weekday',
                         'fecha_poliza_vto_natural_year',
                         'fecha_poliza_vto_natural_month',
                         'fecha_poliza_vto_natural_day',
                         'fecha_poliza_vto_natural_weekday',
                         'fecha_siniestro_ocurrencia_year',
                         'fecha_siniestro_ocurrencia_month',
                         'fecha_siniestro_ocurrencia_day',
                         'fecha_siniestro_ocurrencia_weekday',
                         'fecha_primera_visita_peritaje_year',
                         'fecha_primera_visita_peritaje_month',
                         'fecha_primera_visita_peritaje_day',
                         'fecha_primera_visita_peritaje_weekday',
                         'fecha_ultima_visita_peritaje_year',
                         'fecha_ultima_visita_peritaje_month',
                         'fecha_ultima_visita_peritaje_day',
                         'fecha_ultima_visita_peritaje_weekday',
                         'fecha_siniestro_comunicación_year',
                         'fecha_siniestro_comunicación_month',
                         'fecha_siniestro_comunicación_day',
                         'fecha_siniestro_comunicación_weekday',
                         'id_poliza', 'hogar_poblacion', 'version_poliza'
                         ]
        df = df.drop(*del_variables)

        # FECHAS LOGICAS: We create different types of dates var that can be relevant to fraud analysis.
        # Diferencia entre primera póliza emisión y último vencimiento natural
        df = df.withColumn('fecha_diferencia_vto_emision', datediff(df['fecha_poliza_vto_natural'],
                                                                    df['fecha_poliza_emision']))

        # if fecha efecto < fecha emision => d = 1
        df = df.withColumn('fecha_indicador_efecto_emision', when(df['fecha_poliza_emision'] > df[
            'fecha_poliza_efecto_natural'], 1).otherwise(0))

        # diferencia entre siniestro y efecto: 5, 15, 30 días
        df = df.withColumn('fecha_diferencia_siniestro_efecto',
                           datediff(df['fecha_siniestro_ocurrencia'], df['fecha_poliza_efecto_natural']))
        days_var = [5, 15, 30]
        for col in days_var:
            df = df.withColumn('fecha_diferencia_siniestro_efecto_' + str(col),
                               when(df['fecha_diferencia_siniestro_efecto'] <= col, 1).otherwise(0))

        # diferencia entre siniestro y primera emisión: 5, 15, 30 días
        df = df.withColumn('fecha_diferencia_siniestro_emision', datediff(df['fecha_siniestro_ocurrencia'],
                                                                          df['fecha_poliza_emision']))
        for col in days_var:
            df = df.withColumn('fecha_diferencia_siniestro_emision_' + str(col),
                               when(df['fecha_diferencia_siniestro_emision'] <= col, 1).otherwise(0))

        # diferencia entre siniestro y vencimiento 5, 15, 30 días
        df = df.withColumn('fecha_diferencia_siniestro_vto_natural', datediff(df['fecha_poliza_vto_natural'],
                                                                              df['fecha_siniestro_ocurrencia']))
        for col in days_var:
            df = df.withColumn('fecha_diferencia_siniestro_vto_natural_' + str(col),
                               when(df['fecha_diferencia_siniestro_vto_natural'] <= col, 1).otherwise(0))

        # if fecha comunicacion > fecha ocurrencia en 7 días, d = 1
        df = df.withColumn('fecha_diferencia_siniestro_comunicacion', datediff(df['fecha_siniestro_comunicacion'],
                                                                               df['fecha_siniestro_ocurrencia']))
        df = df.withColumn('fecha_diferencia_comunicacion_outlier',
                           when(df['fecha_diferencia_siniestro_comunicacion'] >= 7, 1).otherwise(0))
        df = df.drop('fecha_siniestro_comunicacion')

        df = df.dropDuplicates(subset=['id_siniestro'])

        return df

    def load_data(self, df):
        """Collect data locally and write to CSV.
        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.fecha_output_prediction
        else:
            name = STRING.fecha_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


if __name__ == '__main__':
    Fecha(is_diario=False).run()
