import datetime
import sys

from pyspark.sql.functions import when, udf, regexp_replace, sum as sum_, lit, datediff
from pyspark.sql.types import IntegerType, DateType, FloatType, StructType, StructField, StringType
from pyspark.sql.window import Window

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import outliers


class Peritaje(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("PeritajeTask")

    def run(self):
        print("Running Peritaje")
        df, bl, peritos_aux, peritos_nif_aux, reparador_aux = self.extract_data()
        df, peritos_aux, peritos_nif_aux, reparador_aux = self.transform_data(df,
                                                                              bl_processed=bl,
                                                                              peritos_aux=peritos_aux,
                                                                              perito_nif_aux=peritos_nif_aux,
                                                                              reparador_aux=reparador_aux)
        self.load_data(df, peritos_aux, peritos_nif_aux, reparador_aux)


    def extract_data(self):
        """Load data from Parquet file format.

        :param spark: Spark session object.
        :param file: File name as input
        :param diario_: If it is daily or montlhy process
        :return: Spark DataFrame.
        """
        if self._is_diario:
            df = (
                self._spark
                .read
                .csv(STRING.peritaje_input_prediction, header=True, sep=',', nullValue='?'))

            peritos_aux = (
                self._spark
                .read
                .csv(STRING.peritaje_output_aux_perito_nombre_training, header=True, sep=',', nullValue='?')
            )

            peritos_nif_aux = (
                self._spark
                .read
                .csv(STRING.peritaje_output_aux_perito_nif_training, header=True, sep=',', nullValue='?')
            )

            reparador_aux = (
                self._spark
                .read
                .csv(STRING.peritaje_output_aux_perito_reparador_training, header=True, sep=',', nullValue='?')
            )

        else:
            df = (
                self._spark
                .read
                .csv(STRING.peritaje_input_training, header=True, sep=',', nullValue='?'))

            peritos_aux = None
            peritos_nif_aux = None
            reparador_aux = None

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

        bl_processed = (self._spark.
                        read.
                        csv(STRING.reporting_output, sep=';',
                            header=True,
                            encoding='UTF-8', schema=custom_schema))

        return df, bl_processed, peritos_aux, peritos_nif_aux, reparador_aux



    def transform_data(self, df, bl_processed, peritos_aux, perito_nif_aux, reparador_aux):
        """Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))
        df = df.withColumn('peritaje_transferencia_IBAN', regexp_replace('peritaje_transferencia_IBAN', ' ', ''))
        df = df.fillna({'peritaje_transferencia_IBAN': 0})
        df = df.withColumn('peritaje_transferencia_IBAN', df['peritaje_transferencia_IBAN'].cast('float'))
        df = df.withColumn('peritaje_transferencia_nif', df['peritaje_transferencia_nif'].cast('string'))
        df = df.withColumn('peritaje_codigo', df['peritaje_codigo'].cast(IntegerType()))

        # VARIABLES DE FECHA
        fecha_variables = ['fecha_encargo_peritaje', 'peritaje_fecha_informe']
        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())
        for col in fecha_variables:
            df = df.fillna({col: '1900/01/01'})
            df = df.withColumn(col, func(df[col]))
            df = df.withColumn(col, when(df[col] == '1900-01-01', None).otherwise(df[col]))

        # Sort Values by Perito-Visita
        df = df.orderBy(['id_siniestro', 'peritaje_codigo'], ascending=[True, True])

        # INFORME DEFINITIVO exist: First we count the null informe_definitivo. Then we cumulate the values.
        # Finally, we replace positive values with one, because we just want to identify.
        df = df.withColumn('peritaje_no_informe_definitivo', when(df['peritaje_fecha_informe'].isNull(), 1).otherwise(0))
        w = (Window().partitionBy(df.id_siniestro).rowsBetween(-sys.maxsize, sys.maxsize))

        df = df.withColumn('peritaje_no_informe_definitivo', sum_(df.peritaje_no_informe_definitivo).over(w))
        df = df.withColumn('peritaje_no_informe_definitivo', when(df['peritaje_no_informe_definitivo'] > 0, 1).otherwise(0))

        # COUNT PERITAJES: We count the adjust-loss visits using the number of times is repeated a sinister.
        df = df.withColumn('peritaje_pondera', lit(1))
        df = df.withColumn('peritaje_count', sum_(df.peritaje_pondera).over(w))

        # FECHA PRIMER PERITAJE - ULTIMO PERITAJE: To each sinister we just keep the first date of the adjust-loss
        # visit and the last date.
        # First Date: We create a table that only keep the sinister and the first date. Then we remarge keeping the
        # first value of each sinister
        fecha_primer_df = df.select(['id_siniestro', 'fecha_encargo_peritaje'])
        fecha_primer_df = fecha_primer_df.orderBy(['id_siniestro', 'fecha_encargo_peritaje'], ascending=[True, False])
        fecha_primer_df = fecha_primer_df.dropDuplicates(subset=['id_siniestro'])
        fecha_primer_df = fecha_primer_df.withColumnRenamed('fecha_encargo_peritaje', 'fecha_primer_peritaje')
        df = df.join(fecha_primer_df, on='id_siniestro', how='left')
        del fecha_primer_df

        fecha_ultimo_df = df.select(['id_siniestro', 'peritaje_fecha_informe'])
        fecha_ultimo_df = fecha_ultimo_df.orderBy(['id_siniestro', 'peritaje_fecha_informe'], ascending=[True, False])
        fecha_ultimo_df = fecha_ultimo_df.dropDuplicates(subset=['id_siniestro'])
        fecha_ultimo_df = fecha_ultimo_df.withColumnRenamed('peritaje_fecha_informe', 'fecha_ultimo_informe')
        df = df.join(fecha_ultimo_df, on='id_siniestro', how='left')
        del fecha_ultimo_df

        # TIEMPO ENTRE PERITAJES: fecha_ultimo_informe - fecha_primer_peritaje
        df = df.withColumn('peritaje_duracion', datediff(df['fecha_ultimo_informe'], df['fecha_primer_peritaje']))

        # DURACION PROMEDIO PERITAJE: We calculate the duration time average by number of visits in each sinister
        df = df.withColumn('peritaje_duracion_promedio', df['peritaje_duracion'] / df['peritaje_count'])

        # FECHA_INFORME EXTENDIDA: We convert fecha_informe_encargo_extendida into two values S = 1, N = 0
        df = df.withColumn('fecha_informe_encargo_extendida',
                           when(df['fecha_informe_encargo_extendida'] == 'S', 1).otherwise(0))
        df = df.withColumn('fecha_informe_encargo_extendida', sum_(df.fecha_informe_encargo_extendida).over(w))

        # CATEGORICAL VAR: Using dummy variables, we generate the cumsum and the average of each value. 'count' will
        # count the total value summing each survey visit value. 'promedio' will take the average of surveys in
        # each sinister
        var_dummies = ['peritaje_pregunta_1', 'peritaje_pregunta_2',
                       'peritaje_pregunta_3', 'peritaje_pregunta_4',
                       'peritaje_pregunta_5', 'peritaje_pregunta_6a',
                       'peritaje_pregunta_7', 'peritaje_pregunta_8a',
                       'peritaje_pregunta_9', 'peritaje_pregunta_10',
                       'peritaje_pregunta_11', 'peritaje_pregunta_12',
                       'peritaje_pregunta_13', 'peritaje_pregunta_14',
                       'peritaje_pregunta_15', 'peritaje_pregunta_16',
                       'peritaje_pregunta_17', 'peritaje_pregunta_18',
                       'peritaje_pregunta_19', 'peritaje_pregunta_20',
                       'peritaje_negativo', 'peritaje_posible_fraude',
                       'peritaje_negativos_perito'
                       ]

        for col in var_dummies:
            # We generate a table dummy so we can drop the nan values and therefore we do not take into account
            # this values on sum and average value
            dummie_sum = df.select(['id_siniestro', col])

            # We drop the nan values in each loss adjuster visit
            dummie_sum = dummie_sum.dropna(subset=[col])

            # Now we count how many surveys were made
            dummie_sum = dummie_sum.withColumn('peritaje_pondera', lit(1))
            w_dummy = (Window().partitionBy(dummie_sum.id_siniestro).rowsBetween(-sys.maxsize, sys.maxsize))
            dummie_sum = dummie_sum.withColumn('peritaje_count', sum_(dummie_sum.peritaje_pondera).over(w_dummy))

            # We take the total sum and the average
            name = col + '_count'
            promedio = col + '_promedio'
            if col != 'peritaje_negativos_perito':
                dummie_sum = dummie_sum.withColumn(col, dummie_sum[col].cast(IntegerType()))
            else:
                dummie_sum = dummie_sum.withColumn(col, dummie_sum[col].cast('float'))

            dummie_sum = dummie_sum.withColumn(name, sum_(dummie_sum[col]).over(w_dummy))
            dummie_sum = dummie_sum.withColumn(promedio, dummie_sum[name] / dummie_sum['peritaje_count'])

            # We delete peritaje_count and pondera so we do not have problem with the merge
            dummie_sum = dummie_sum.drop(*[col, 'peritaje_count', 'peritaje_pondera'])

            # We drop the duplicates values in each siniester and keep the last which is the value that cumsum the total
            # values
            dummie_sum = dummie_sum.dropDuplicates(subset=['id_siniestro'])

            # We merge the table with the original dataframe
            df = df.join(dummie_sum, on='id_siniestro', how='left')
            del dummie_sum

        # PERITAJE_NEGATIVO: If peritaje_negativo_count > 0 => peritaje_negativo = 1
        df = df.withColumn('peritaje_negativo', when(df['peritaje_negativo_count'] > 0, 1).otherwise(0))

        # PERITAJE NEGATIVO PERITO: We made delete the sum count, we just need the average
        df = df.drop('peritaje_negativos_perito_count')

        # IBAN DE PERITO: We check if the adjust losser's IBAN is registred in the blacklist
        bl_processed_iban = bl_processed.filter(~((bl_processed['iban'].isNull()) | (bl_processed['iban'] == '?')))
        bl_processed_iban = bl_processed_iban.select('iban')
        bl_processed_iban = bl_processed_iban.dropDuplicates(subset=['iban'])
        df = df.join(bl_processed_iban, df.peritaje_transferencia_IBAN == bl_processed_iban.iban, how='left')
        df = df.withColumn('peritaje_iban_blacklist', when(df['iban'].isNull(), 0).otherwise(1))

        # PERITAJE INDEM INFORME PREVIO: We sum the total values
        df = df.withColumn('peritaje_indem_informe_previo', df['peritaje_indem_informe_previo'].cast('float'))
        df = df.withColumn('peritaje_indem_informe_previo_sum', sum_(df['peritaje_indem_informe_previo']).over(w))

        # PERIAJTE_INDEM_INFORME_DEFNITIVO: We sum the total values
        df = df.withColumn('peritaje_indem_informe_definitivo', df['peritaje_indem_informe_definitivo'].cast('float'))
        df = df.withColumn('peritaje_indem_informe_definitivo_sum', sum_(df['peritaje_indem_informe_definitivo']).over(w))

        # DIFERENCIA DEFINITIVO-PREVIO
        df = df.withColumn('peritaje_indem_previo_definitivo_sum', df['peritaje_indem_informe_definitivo_sum'] -
                           df['peritaje_indem_informe_previo_sum'])

        # COBERTURAS: valores extraños
        df = df.withColumn('peritaje_coberturas_indemnizar_previo',
                           df['peritaje_coberturas_indemnizar_previo'].cast(IntegerType()))
        df = df.withColumn('peritaje_coberturas_indemnizar_MIGRA',
                           when(df['peritaje_coberturas_indemnizar_previo'] > 7, 1).otherwise(0))
        df = df.withColumn('peritaje_coberturas_indemnizar_previo', when(df['peritaje_coberturas_indemnizar_previo'] > 10,
                                                                         df['peritaje_coberturas_indemnizar_definitivo']
                                                                         ).otherwise(
            df['peritaje_coberturas_indemnizar_previo']))

        # COBERTURAS INFORME PREVIO
        df = df.withColumn('peritaje_coberturas_indemnizar_previo',
                           df['peritaje_coberturas_indemnizar_previo'].cast('float'))
        df = df.withColumn('peritaje_coberturas_indemnizar_previo_sum',
                           sum_(df.peritaje_coberturas_indemnizar_previo).over(w))
        df = df.withColumn('peritaje_coberturas_indemnizar_previo_promedio',
                           df['peritaje_coberturas_indemnizar_previo_sum'] / df['peritaje_count'])

        # COBERTURAS INFORME DEFINITIVO
        df = df.withColumn('peritaje_coberturas_indemnizar_definitivo',
                           df['peritaje_coberturas_indemnizar_definitivo'].cast('float'))
        df = df.withColumn('peritaje_coberturas_indemnizar_definitivo_sum',
                           sum_(df.peritaje_coberturas_indemnizar_definitivo).over(w))
        df = df.withColumn('peritaje_coberturas_indemnizar_definitivo_promedio',
                           df['peritaje_coberturas_indemnizar_definitivo_sum'] / df['peritaje_count'])

        # DIFERENCIA COBERTURAS DEFINITIVO-PREVIO
        df = df.withColumn('peritaje_cobertura_previo_definitivo_sum', df['peritaje_coberturas_indemnizar_definitivo_sum'] -
                           df['peritaje_coberturas_indemnizar_previo_sum'])

        # PORCENTUAL VARIATION INDEM-COBERTURAS: Here we make the difference betweeen the last visit. So we do not have
        # to acumulate the values. We take the variation and, at the end, we keep the last row. We have to normalize to
        # avoid the zero division problem. For COBERTURAS, due it is an int value we can get the % value for zero
        # division with the diff. However, for INDEM, it can generate extra large values, so we left it in Zero.

        funct_indem = udf(lambda x_definitivo, x_previo: x_definitivo / x_previo - 1 if x_previo != 0 else 0., FloatType())
        funct_garan = udf(lambda x_definitivo, x_previo,
                          x_final_inicial: x_definitivo / x_previo - 1 if x_previo != 0 else x_final_inicial,
                          FloatType())

        df = df.withColumn('peritaje_indem_final_inicial',
                           funct_indem(df.peritaje_indem_informe_definitivo, df.peritaje_indem_informe_previo))
        df = df.withColumn('peritaje_garantia_final_inicial',
                           funct_garan(df.peritaje_coberturas_indemnizar_definitivo,
                                       df.peritaje_coberturas_indemnizar_previo, df.peritaje_garantia_final_inicial))
        df = df.withColumn('peritaje_indem_final_inicial',
                           funct_garan(df.peritaje_coberturas_indemnizar_definitivo,
                                       df.peritaje_coberturas_indemnizar_previo, df.peritaje_garantia_final_inicial))

        # PERITO EN BLACKLIST POR NIF: We count how many times the loss-adjuster NIF appears in the historic blacklist
        bl_processed = bl_processed.select('nif_o_intm')
        bl_processed = bl_processed.dropDuplicates(subset=['nif_o_intm'])
        df = df.join(bl_processed, df.peritaje_transferencia_nif == bl_processed.nif_o_intm, how='left')
        df = df.withColumn('perito_fraud_claims', when(df['nif_o_intm'].isNull(), 0).otherwise(1))

        perito_base = df.select(['id_siniestro', 'peritaje_nombre', 'peritaje_transferencia_nif',
                                 'peritaje_aide_nombre_reparador', 'peritaje_pondera', 'peritaje_negativo',
                                 'peritaje_indem_informe_definitivo', 'peritaje_coberturas_indemnizar_definitivo'])

        df = df.dropDuplicates(subset='id_siniestro')

        # TABLA DE PERITOS BY AGRUPACION: Some statistics about the loss-adjuster only for the current DB
        if self._is_prediction:
            df = df.join(peritos_aux, on='peritaje_nombre', how='left')
            peritos_cols = list(peritos_aux.columns)
            peritos_cols.remove('peritaje_nombre')
            for col in peritos_cols:
                df = df.fillna({col: 0})
            del peritos_cols

        else:
            peritos = perito_base.select(['id_siniestro', 'peritaje_nombre',
                                          'peritaje_pondera', 'peritaje_negativo',
                                          'peritaje_indem_informe_definitivo', 'peritaje_coberturas_indemnizar_definitivo'])

            # paso 1: Eliminar NaN
            peritos = peritos.dropna(subset='peritaje_nombre')

            # How many visits by PERITAJE_NOMBRE
            w_peritos = (Window().partitionBy(peritos.peritaje_nombre).rowsBetween(-sys.maxsize, sys.maxsize))
            peritos = peritos.withColumn('peritaje_count_visitasxperito', sum_(df.peritaje_pondera).over(w_peritos))

            # How many negative claims by PERITAJE_NOMBRE
            peritos_temp = peritos.filter(~peritos['peritaje_negativo'].isNull())
            peritos_temp = peritos_temp.withColumn('peritaje_negativo', df['peritaje_negativo'].cast(IntegerType()))
            w_peritos_temp = (Window().partitionBy('peritaje_nombre').rowsBetween(-sys.maxsize, sys.maxsize))
            peritos_temp = peritos_temp.withColumn('peritaje_count_negativoxperito',
                                                   sum_(df.peritaje_negativo).over(w_peritos_temp))
            peritos_temp = peritos_temp.fillna({'peritaje_count_negativoxperito': 0})
            peritos_temp = peritos_temp.select(['peritaje_nombre', 'peritaje_count_negativoxperito'])
            peritos_temp = peritos_temp.dropDuplicates(subset='peritaje_nombre')
            peritos = peritos.join(peritos_temp, on='peritaje_nombre', how='left')
            del peritos_temp, w_peritos_temp

            peritos = peritos.drop('peritaje_negativo')

            # Average of Negative Surveys by PERITAJE_NOMBRE
            peritos = peritos.withColumn('peritaje_perito_porc_negativos', peritos['peritaje_count_negativoxperito']
                                         / peritos['peritaje_count_visitasxperito'])

            # Amount of Indem definitiva by PERITAJE_NOMBRE
            peritos = peritos.withColumn('peritaje_sum_indem_definitivaxperito',
                                         sum_('peritaje_indem_informe_definitivo').over(w_peritos))
            peritos = peritos.drop('peritaje_indem_informe_definitivo')

            # Amount of Coberturas by PERITAJE_ NOMBRE
            peritos = peritos.withColumn('peritaje_sum_coberturaxperito',
                                         sum_('peritaje_coberturas_indemnizar_definitivo').over(w_peritos))
            peritos = peritos.drop('peritaje_coberturas_indemnizar_definitivo')

            # paso 2: Mantenemos el último siniestro para poder agrupar el fraude por siniestro
            peritos = peritos.dropDuplicates(subset=['id_siniestro', 'peritaje_nombre'])

            # How many claims by PERITAJE_NOMBRE
            w_peritos = (Window().partitionBy(peritos.peritaje_nombre).rowsBetween(-sys.maxsize, sys.maxsize))
            peritos = peritos.withColumn('peritaje_count_siniestrosxperito', sum_(df.peritaje_pondera).over(w_peritos))
            peritos = peritos.drop('peritaje_pondera')

            # Average of Indem definitiva by PERITAJE_NOMBRE
            peritos = peritos.withColumn('peritaje_promedio_indem_definitivaxperito',
                                         peritos['peritaje_sum_indem_definitivaxperito'] / peritos[
                                             'peritaje_count_siniestrosxperito'])

            # We merge with the known cases of Fraud
            file_blacklist_resume = bl_processed.select('id_siniestro')
            file_blacklist_resume = file_blacklist_resume.dropDuplicates(subset='id_siniestro')
            file_blacklist_resume = file_blacklist_resume.withColumn('peritos_fraud', lit(1))
            peritos = peritos.join(file_blacklist_resume, on='id_siniestro', how='left')
            peritos = peritos.fillna({'peritos_fraud': 0})

            # How many fraud sinister by PERITAJE_NOMBRE
            peritos = peritos.withColumn('peritaje_count_FRAUDxperito', sum_(peritos['peritos_fraud']).over(w_peritos))

            # Average fraud sinister by PERITAJE_NOMBRE
            peritos = peritos.withColumn('peritaje_promedio_FRAUDxsiniestroxperito',
                                         peritos['peritaje_count_FRAUDxperito'] / peritos[
                                             'peritaje_count_siniestrosxperito'])

            # We get the values by PERITAJE_NOMBRE and calculate
            # paso 3: mantenemos solo al perito
            peritos_temp = peritos.dropDuplicates(subset=['peritaje_nombre'])
            peritos_temp = outliers.Outliers.outliers_mad(peritos_temp, 'peritaje_promedio_FRAUDxsiniestroxperito',
                                                          not_count_zero=False)
            peritos_temp = outliers.Outliers.outliers_mad(peritos_temp, 'peritaje_promedio_indem_definitivaxperito',
                                                          not_count_zero=False)
            peritos_temp = outliers.Outliers.outliers_mad(peritos_temp, 'peritaje_sum_coberturaxperito',
                                                          not_count_zero=False)
            peritos_temp = peritos_temp.drop('id_siniestro')
            peritos = peritos.select(['id_siniestro', 'peritaje_nombre'])
            peritos = peritos.join(peritos_temp, on='peritaje_nombre', how='left')
            del peritos_temp

            peritos = peritos.dropDuplicates(subset=['peritaje_nombre'])
            peritos = peritos.drop('id_siniestro')

            df = df.join(peritos, on='peritaje_nombre', how='left')
            peritos_cols = list(peritos.columns)
            peritos_cols.remove('peritaje_nombre')
            for col in peritos_cols:
                df = df.fillna({col: 0})
            del peritos_cols

        # TABLA DE PERITOS BY AGRUPACION: Some statistics about the loss-adjuster only for the current DB
        if self._is_prediction:
            df = df.join(perito_nif_aux, how='left', on='peritaje_transferencia_nif')
            peritos_cols = perito_nif_aux.columns.values.tolist()
            peritos_cols.remove('peritaje_transferencia_nif')
            for col in peritos_cols:
                df = df.fillna({col: 0})
            del peritos_cols

        else:
            perito_nif = perito_base.select(['id_siniestro', 'peritaje_transferencia_nif',
                                             'peritaje_pondera', 'peritaje_negativo',
                                             'peritaje_indem_informe_definitivo',
                                             'peritaje_coberturas_indemnizar_definitivo'])

            # paso1: Eliminar NaN
            perito_nif = perito_nif.dropna(subset='peritaje_transferencia_nif')

            # How many visits by PERITAJE_NOMBRE
            w_peritos = (Window().partitionBy(perito_nif.peritaje_transferencia_nif).rowsBetween(-sys.maxsize, sys.maxsize))
            perito_nif = perito_nif.withColumn('peritaje_count_visitasxnif',
                                               sum_(perito_nif['peritaje_pondera']).over(w_peritos))

            # How many negative claims by PERITAJE_NIF
            peritos_temp = perito_nif.filter(~perito_nif['peritaje_negativo'].isNull())
            w_peritos_temp = (
                Window().partitionBy(peritos_temp.peritaje_transferencia_nif).rowsBetween(-sys.maxsize, sys.maxsize))
            peritos_temp = peritos_temp.withColumn('peritaje_negativo',
                                                   peritos_temp['peritaje_negativo'].cast(IntegerType()))
            peritos_temp = peritos_temp.withColumn('peritaje_nif_count_negativoxperito',
                                                   sum_(peritos_temp['peritaje_negativo']).over(w_peritos_temp))
            peritos_temp = peritos_temp.fillna({'peritaje_nif_count_negativoxperito': 0})
            peritos_temp = peritos_temp.select(['peritaje_transferencia_nif', 'peritaje_nif_count_negativoxperito'])
            peritos_temp = peritos_temp.dropDuplicates(subset=['peritaje_transferencia_nif'])
            perito_nif = perito_nif.join(peritos_temp, on='peritaje_transferencia_nif', how='left')
            del peritos_temp, w_peritos_temp

            perito_nif = perito_nif.drop('peritaje_negativo')

            # Average of Negative Surveys by PERITAJE_NIF
            perito_nif = perito_nif.withColumn('peritaje_nif_perito_porc_negativos',
                                               perito_nif['peritaje_nif_count_negativoxperito']
                                               / perito_nif['peritaje_count_visitasxnif'])

            # Amount of Indem definitiva by PERITAJE_NIF
            perito_nif = perito_nif.withColumn('peritaje_indem_informe_definitivo',
                                               perito_nif['peritaje_indem_informe_definitivo'].cast(FloatType()))
            perito_nif = perito_nif.withColumn('peritaje_nif_sum_indem_definitivaxperito',
                                               sum_(perito_nif['peritaje_indem_informe_definitivo']).over(w_peritos))
            perito_nif = perito_nif.drop('peritaje_indem_informe_definitivo')

            # Amount of Coberturas by PERITAJE_ NIF
            perito_nif = perito_nif.withColumn('peritaje_nif_sum_coberturaxperito',
                                               sum_(perito_nif['peritaje_coberturas_indemnizar_definitivo']).over(
                                                   w_peritos))
            peritos_nif = perito_nif.drop('peritaje_coberturas_indemnizar_definitivo')

            # paso 2: Mantenemos el último siniestro para poder agrupar el fraude por siniestro
            # How many claims by PERITAJE_NIF
            perito_nif = perito_nif.withColumn('peritaje_nif_count_siniestrosxperito',
                                               sum_(perito_nif['peritaje_pondera']).over(w_peritos))
            perito_nif = perito_nif.drop('peritaje_pondera')

            # Average of Indem definitiva by PERITAJE_NIF
            perito_nif = perito_nif.withColumn('peritaje_nif_promedio_indem_definitivaxperito',
                                               perito_nif['peritaje_nif_sum_indem_definitivaxperito'] / perito_nif[
                                                   'peritaje_nif_count_siniestrosxperito'])

            # Average of Coberturas by PERITAJE_ NIF
            perito_nif = perito_nif.withColumn('peritaje_nif_promedio_coberturaxperito',
                                               perito_nif['peritaje_nif_sum_coberturaxperito'] / perito_nif[
                                                   'peritaje_nif_count_siniestrosxperito'])

            perito_nif = perito_nif.dropDuplicates(subset=['id_siniestro', 'peritaje_transferencia_nif'])

            # We merge with the known cases of Fraud
            file_blacklist_resume = file_blacklist_resume.select('id_siniestro')
            file_blacklist_resume = file_blacklist_resume.withColumn('perito_nif_fraud', lit(1))
            perito_nif = perito_nif.join(file_blacklist_resume, on='id_siniestro', how='left')
            perito_nif = perito_nif.fillna({'perito_nif_fraud': 0})

            # How many fraud sinister by PERITAJE_NIF
            perito_nif = perito_nif.withColumn('peritaje_nif_count_FRAUDxperito',
                                               sum_(perito_nif['peritos_fraud']).over(w_peritos))
            perito_nif = perito_nif.drop('peritos_fraud')

            # Average fraud sinister by PERITAJE_NIF
            perito_nif = perito_nif.withColumn('peritaje_nif_promedio_FRAUDxsiniestroxperito',
                                               perito_nif['peritaje_nif_count_FRAUDxperito'] / perito_nif[
                                                   'peritaje_nif_count_siniestrosxperito'])

            # We get the values by PERITAJE_NIF and calculate
            peritos_temp = perito_nif.dropDuplicates(subset=['peritaje_transferencia_nif'])
            peritos_temp = peritos_temp.drop('id_siniestro')
            peritos_temp = outliers.Outliers.outliers_mad(peritos_temp, 'peritaje_nif_promedio_FRAUDxsiniestroxperito',
                                                          not_count_zero=False)
            peritos_temp = outliers.Outliers.outliers_mad(peritos_temp, 'peritaje_nif_promedio_indem_definitivaxperito',
                                                          not_count_zero=False)
            peritos_temp = outliers.Outliers.outliers_mad(peritos_temp, 'peritaje_nif_sum_coberturaxperito',
                                                          not_count_zero=False)
            peritos_nif = peritos_nif.select(['id_siniestro', 'peritaje_transferencia_nif'])
            peritos_nif = peritos_nif.join(peritos_temp, on='peritaje_transferencia_nif', how='left')
            del peritos_temp

            peritos_nif = peritos_nif.dropDuplicates(subset=['peritaje_transferencia_nif'])
            peritos_nif = peritos_nif.drop('id_siniestro')

            # del peritos['peritaje_transferencia_nif']
            df = df.join(peritos_nif, on='peritaje_transferencia_nif', how='left')
            peritos_cols = peritos_nif.columns.values.tolist()
            peritos_cols.remove('peritaje_transferencia_nif')
            for col in peritos_cols:
                df = df.fillna({col: 0})
            del peritos_cols

        # TABLA REPARADOR: Some statistics based on the current database
        if self._is_prediction:
            df = df.join(reparador_aux, how='left', on='peritaje_aide_nombre_reparador')
            reparador_cols = reparador_aux.columns.values.tolist()
            reparador_cols.remove('peritaje_aide_nombre_reparador')
            for col in reparador_cols:
                df = df.fillna({col: 0})

        else:
            reparador = perito_base.select([['id_siniestro', 'peritaje_aide_nombre_reparador', 'peritaje_pondera']])
            del perito_base

            reparador = reparador.dropna(subset='peritaje_aide_nombre_reparador')
            reparador = reparador.dropDuplicates(subset=['peritaje_aide_nombre_reparador', 'id_siniestro'])

            # We merge with the known cases of Fraud
            file_blacklist_resume = file_blacklist_resume.select('id_siniestro')
            file_blacklist_resume = file_blacklist_resume.withColumn('reparador_fraud', lit(1))
            reparador = reparador.join(file_blacklist_resume, on='id_siniestro', how='left')
            reparador = reparador.fillna({'reparador_fraud': 0})

            w_reparador = (
                Window().partitionBy(reparador.peritaje_aide_nombre_reparador).rowsBetween(-sys.maxsize, sys.maxsize))

            reparador = reparador.withColumn('peritaje_count_visitasxreparador',
                                             sum_(reparador['peritaje_pondera']).over(w_reparador))
            reparador = reparador.drop('peritaje_pondera')

            reparador = reparador.withColumn('peritaje_count_FRAUDxreparador',
                                             sum_(df['reparador_fraud']).over(w_reparador))
            reparador = reparador.drop('reparador_fraud')
            reparador = reparador.withColumn('peritaje_promedio_FRAUDxsiniestroxreparador',
                                             reparador['peritaje_count_FRAUDxreparador'] / reparador[
                                                 'peritaje_count_visitasxreparador'])

            reparador = reparador.dropDuplicates(subset=['peritaje_aide_nombre_reparador'])
            reparador = outliers.Outliers.outliers_mad(reparador, 'peritaje_promedio_FRAUDxsiniestroxreparador',
                                                       justnot_count_zero=True)
            reparador = reparador.drop('id_siniestro')

            df = df.join(reparador, how='left', on='peritaje_aide_nombre_reparador')
            reparador_cols = reparador.columns.values.tolist()
            reparador_cols.remove('peritaje_aide_nombre_reparador')
            for col in reparador_cols:
                df = df.fillna({col: 0})
            del reparador_cols

        # DELETE VARIABLES
        del_variables = ["peritaje_siniestro_causa", "peritaje_informe_previo_reserva_estimada",
                         'peritaje_informe_previo_origen_siniestro', 'peritaje_pregunta_8b',
                         'peritaje_pregunta_6b', "peritaje_obs_documentacion", "peritaje_obs_fotos",
                         'id_poliza', 'version_poliza', 'peritaje_codigo', 'fecha_encargo_peritaje',
                         'peritaje_numero_informe', 'peritaje_fecha_informe', 'fecha_informe_encargo_extendida',
                         'peritaje_nombre', 'peritaje_aide_nombre_reparador', 'peritaje_pregunta_1',
                         'peritaje_pregunta_2', 'peritaje_pregunta_3', 'peritaje_pregunta_4',
                         'peritaje_pregunta_5', 'peritaje_pregunta_6a',
                         'peritaje_pregunta_7', 'peritaje_pregunta_8a',
                         'peritaje_pregunta_9', 'peritaje_pregunta_10',
                         'peritaje_pregunta_11', 'peritaje_pregunta_12',
                         'peritaje_pregunta_13', 'peritaje_pregunta_14',
                         'peritaje_pregunta_15', 'peritaje_pregunta_16',
                         'peritaje_pregunta_17', 'peritaje_pregunta_18',
                         'peritaje_pregunta_19', 'peritaje_pregunta_20', 'peritaje_posible_fraude',
                         'peritaje_negativo', 'peritaje_transferencia_IBAN', 'peritaje_transferencia_nif',
                         'peritaje_indem_informe_previo', 'peritaje_indem_informe_definitivo',
                         'peritaje_pondera', 'id_fiscal', 'peritaje_negativos_perito',
                         'peritaje_coberturas_indemnizar_previo', 'peritaje_coberturas_indemnizar_definitivo',
                         'audit_siniestro_producto_tecnico',
                         'audit_siniestro_codigo_compania'

                         ]

        df = df.drop(*del_variables)

        return df, peritos, peritos_nif, reparador

    def load_data(self, df, peritos_aux, peritos_nif_aux, reparador_aux):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :param self._is_diario: If it is daily or montlhy process
        :param peritos_aux
        :param peritos_nif_aux
        :param reparador_aux
        :return: None
        """
        if self._is_diario:
            name = STRING.peritaje_output_prediction

        else:
            name = STRING.peritaje_output_training

            (peritos_aux
             .coalesce(1)
             .write
             .csv(STRING.peritaje_output_aux_perito_nombre_training, mode='overwrite', header=True,
                  sep=';'))

            (peritos_nif_aux
             .coalesce(1)
             .write
             .csv(STRING.peritaje_output_aux_perito_nif_training, mode='overwrite', header=True, sep=';'))

            (reparador_aux
             .coalesce(1)
             .write
             .csv(STRING.peritaje_output_aux_perito_reparador_training, mode='overwrite', header=True,
                  sep=';'))

        (df
         .toPandas()
         .to_csv(name, header=True, sep=';', index=False))

if __name__ == '__main__':
    Peritaje(True).run()