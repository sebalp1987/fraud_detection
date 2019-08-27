import sys
import os

from pyspark.sql.functions import when, udf, lit, sum as sum_, max as max_, collect_set, size
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, FloatType
from pyspark.sql.window import Window

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import functions as f, outliers
from fraud_home.resources.fraud_home import checklist_spark


class PoReserva(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("PO Reserva")

    def run(self):
        df, df_base, df_processed, id_base, id_new, bl_processed, df_perceptor_aux, df_servicios_aux = self._extract_data()
        df = self._transform_data(df, df_base=df_base, df_processed=df_processed, id_base=id_base, id_new=id_new,
                                  bl_processed=bl_processed,
                                  aux_perceptor=df_perceptor_aux, aux_servicios=df_servicios_aux)
        self._load_data(df)

    def _extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """
        if self._is_diario:
            df = (
                self._spark
                    .read
                    .csv(STRING.poreservable_input_prediction, header=True, sep=',', nullValue='?'))
            df_base = (
                    self._spark
                        .read
                        .csv(STRING.poreservable_input_training, header=True, sep=',', nullValue='?'))

            file_list = [filename for filename in os.listdir(STRING.poreservable_output_training) if
                         filename.endswith('.csv')]

            df_processed = (
                    self._spark
                        .read
                        .csv(STRING.poreservable_output_training + file_list[0], header=True, sep=';', nullValue='?'))

            id_base = (
                    self._spark
                        .read
                        .csv(STRING.id_input_training, header=True, sep=',', nullValue='?'))

            id_new = (
                self._spark
                    .read
                    .csv(STRING.id_input_prediction, header=True, sep=',', nullValue='?'))

            file_list = [filename for filename in os.listdir(STRING.training_auxiliar_perceptor) if
                         filename.endswith('.csv')]
            df_perceptor_aux = (
                self._spark
                    .read
                    .csv(STRING.training_auxiliar_perceptor + file_list[0], header=True, sep=';', nullValue='?')
            )

            file_list = [filename for filename in os.listdir(STRING.training_auxiliar_servicios) if
                         filename.endswith('.csv')]
            df_servicio_aux = (
                self._spark
                    .read
                    .csv(STRING.training_auxiliar_servicios+ file_list[0], header=True, sep=';', nullValue='?')
            )

        else:
            df = (
                self._spark
                    .read
                    .csv(STRING.poreservable_input_training, header=True, sep=',', nullValue='?'))

            df_base = None

            id_base = (
                    self._spark
                        .read
                        .csv(STRING.id_input_training, header=True, sep=',', nullValue='?'))

            df_processed = None

            id_new = None

            df_perceptor_aux = None

            df_servicio_aux = None

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

        return df, df_base, df_processed, id_base, id_new, bl_processed, df_perceptor_aux, df_servicio_aux

    def _transform_data(self, df, df_base, df_processed, id_base, id_new, bl_processed, aux_perceptor, aux_servicios):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param bl_processed
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        if self._is_diario:
            checklist5 = checklist_spark.checklist5(df_reserva=df_base, df_reserva_new=df, df_id=id_base,
                                                    df_id_new=id_new)
        else:
            checklist5 = checklist_spark.checklist5(df_reserva=df, df_id=id_base)

        df = df.join(checklist5, on='id_siniestro', how='left')
        df = df.fillna({'checklist5_poliza': 0, 'checklist5_nif': 0})
        del checklist5

        # RESERVA INICIAL INDEMNIZACION
        reserva_indem = df.select(['id_siniestro', 'po_res_cobertura_id', 'po_res_indem'])
        reserva_indem = reserva_indem.dropDuplicates(subset=['id_siniestro', 'po_res_cobertura_id', 'po_res_indem'])
        reserva_indem = reserva_indem.drop('po_res_cobertura_id')
        reserva_indem = reserva_indem.withColumn('po_res_indem', reserva_indem.po_res_indem.cast(FloatType()))
        reserva_indem = reserva_indem.groupBy(['id_siniestro']).agg(sum_('po_res_indem').alias('po_res_indem'))
        reserva_indem = reserva_indem.withColumn('po_res_indem_mayor_5000',
                                                 when(reserva_indem['po_res_indem'] >= 5000, 1).otherwise(0))
        df = df.drop('po_res_indem')
        df = df.join(reserva_indem, on='id_siniestro', how='left')
        del reserva_indem

        # RESERVA INICIAL GASTO
        reserva_gasto = df.select(['id_siniestro', 'po_res_cobertura_id', 'po_res_gasto'])
        reserva_gasto = reserva_gasto.dropDuplicates(subset=['id_siniestro', 'po_res_cobertura_id', 'po_res_gasto'])
        reserva_gasto = reserva_gasto.drop('po_res_cobertura_id')
        reserva_gasto = reserva_gasto.withColumn('po_res_gasto', reserva_gasto.po_res_gasto.cast(FloatType()))
        reserva_gasto = reserva_gasto.groupBy('id_siniestro').agg(sum_('po_res_gasto').alias('po_res_gasto'))
        reserva_gasto = reserva_gasto.withColumn('po_res_gasto_mayor_1000',
                                                 when(reserva_gasto['po_res_gasto'] >= 1000, 1).otherwise(0))
        reserva_gasto = reserva_gasto.drop('po_res_gasto')
        df = df.join(reserva_gasto, on='id_siniestro', how='left')
        del reserva_gasto

        # COUNT POLIZAS POR SINIESTRO
        df = df.withColumn('pondera_siniestro', lit(1))
        w = (Window().partitionBy('id_siniestro').rowsBetween(-sys.maxsize, sys.maxsize))
        df = df.withColumn('po_reserva_pagoxsiniestro_count', sum_(df['pondera_siniestro']).over(w))
        df = df.withColumn('po_reserva_indemxsiniestro_count', sum_(df['po_pago_indicador_indem']).over(w))
        # PAGO INDEM ANULADOS: Cuando la anulación es == 1 marca tanto el pago como su anulación
        df = df.withColumn('po_pago_indemnizacion_importe_neto',
                           when(df['po_pago_es_anulacion'] == 1, 1).otherwise(df['po_pago_indemnizacion_importe_neto']))

        # GASTO_INDEM_ANULADOS: Cuando el gasto es == 1 marca tanto el pago como su anulación
        df = df.withColumn('po_gasto_indemnizacion_importe_neto',
                           when(df['po_gasto_es_anulacion'] == 1, 1).otherwise(df['po_gasto_es_anulacion']))

        # PAGOS: Sumamos el importe neto de factura + los pagos netos por indemnizaciòn
        df = df.withColumn('po_pago_importe_neto', df['po_pago_factura_importe_neto'].cast(FloatType()) + df[
            'po_pago_indemnizacion_importe_neto'].cast(FloatType()))

        # GASTOS:
        df = df.withColumn('po_gasto_importe_neto', df['po_gasto_factura_importe_neto'].cast(FloatType()) +
                           df['po_gasto_indemnizacion_importe_neto'].cast(FloatType()))

        # PAGO ASEGURADO: Si la persona no es el asegurado ponemos los importes del Asegurado en 0
        df = df.withColumn('po_pago_importe_neto_ASEGURADO',
                           when(df['persona_objeto_asegurado'].cast(IntegerType()) == 0, 0).otherwise(
                               df['po_pago_importe_neto']))

        # IMPORTE PORCENTUAL QUE EFECTIVAMENTE COBRA EL ASEGURADO: importe_neto_asegurado/importe_total
        df = df.withColumn('po_pago_importe_porcentual_ASEGURADO', df['po_pago_importe_neto_ASEGURADO']
                           / (df['po_pago_importe_neto'] + 1))

        # IBAN Blacklist
        bl_processed_iban = bl_processed.filter(~((bl_processed['iban'].isNull()) | (bl_processed['iban'] == '?')))
        bl_processed_iban = bl_processed_iban.select('iban')
        bl_processed_iban = bl_processed_iban.dropDuplicates(subset=['iban'])
        df = df.join(bl_processed_iban, df.po_pago_IBAN == bl_processed_iban.iban, how='left')
        df = df.withColumn('peritaje_pago_iban_blacklist', when(df['iban'].isNull(), 0).otherwise(1))
        df = df.drop('iban')
        df = df.join(bl_processed_iban, df.po_gasto_IBAN == bl_processed_iban.iban, how='left')
        df = df.withColumn('peritaje_gasto_iban_blacklist', when(df['iban'].isNull(), 0).otherwise(1))
        df = df.drop('iban')
        del bl_processed_iban

        # INT Variables
        # Agrupamos los valores INT por Siniestro y lo guardamos en la lista de INT's
        int_var = ['peritaje_pago_iban_blacklist', 'peritaje_gasto_iban_blacklist']
        int_outliers = []

        for col in int_var:
            count = col + '_count'
            df = df.withColumn(count, sum_(df[col]).over(w))
            int_outliers.append(count)
            df = df.drop(col)

        # PERSONA OBJETO RESERVABLE
        df = df.withColumn('id_persona_objeto_reservable_max',
                           max_(df['id_persona_objeto_reservable'].cast(IntegerType())).over(w))
        df = df.drop('id_persona_objeto_reservable')

        # CATEGORICAL VARIABLE
        # Redefinimos la variable pago_gasto_codigo como categórica
        asegurado = STRING.Parameters.Asegurado_Beneficiario_Perjudicado
        profesional = STRING.Parameters.Profesional_Legal
        detective = STRING.Parameters.Detective
        perito = STRING.Parameters.Perito
        reparador = STRING.Parameters.Reparador
        otros = STRING.Parameters.todos

        df = df.withColumn('po_pago_gasto_codigo', df.po_pago_gasto_codigo.cast(StringType()))
        df = df.withColumn('po_pago_gasto_codigo',
                           when(df['po_pago_gasto_codigo'].isin(otros), df['po_pago_gasto_codigo']).otherwise(
                               'Otros'))
        df = df.withColumn('po_pago_gasto_codigo',
                           when(df['po_pago_gasto_codigo'].isin(asegurado),
                                'Asegurado_Beneficiario_Perjudicado').otherwise(df['po_pago_gasto_codigo']))
        df = df.withColumn('po_pago_gasto_codigo',
                           when(df['po_pago_gasto_codigo'].isin(profesional),
                                'Profesional_Legal').otherwise(df['po_pago_gasto_codigo']))
        df = df.withColumn('po_pago_gasto_codigo',
                           when(df['po_pago_gasto_codigo'].isin(detective),
                                'Detective').otherwise(df['po_pago_gasto_codigo']))
        df = df.withColumn('po_pago_gasto_codigo',
                           when(df['po_pago_gasto_codigo'].isin(perito),
                                'Perito').otherwise(df['po_pago_gasto_codigo']))
        df = df.withColumn('po_pago_gasto_codigo',
                           when(df['po_pago_gasto_codigo'].isin(reparador),
                                'Reparador').otherwise(df['po_pago_gasto_codigo']))

        # GARANTIA
        garantia = STRING.Parameters.dict_garantias
        funct = udf(lambda x: f.replace_dict_contain(x, key_values=garantia), StringType())
        df = df.withColumn('po_res_garantia', funct(df['po_res_garantia']))

        # COBERTURA
        cobertura_1 = STRING.Parameters.dict_cobertura_1
        funct = udf(lambda x: f.replace_dict_contain(x, key_values=cobertura_1), StringType())
        df = df.withColumn('po_res_cobertura', funct(df['po_res_cobertura']))
        df = df.withColumn('po_res_cobertura',
                           when(df['po_res_cobertura'] == 'DE', 'ELECTRICIDAD').otherwise(df['po_res_cobertura']))
        cobertura_2 = STRING.Parameters.dict_cobertura_2
        funct = udf(lambda x: f.replace_dict_contain(x, key_values=cobertura_2), StringType())
        df = df.withColumn('po_res_cobertura', funct(df['po_res_cobertura']))

        # Pasamos todas las categóricas para obtener dummies
        categorical_var = ['po_res_garantia', 'po_res_cobertura', 'po_res_situacion',
                           'po_pago_medio', 'po_pago_gasto_codigo']
        variable_dummy = []

        funct = udf(lambda x: f.normalize_string(x), StringType())
        for col in categorical_var:
            df = df.withColumn(col, df[col].cast(StringType()))
            df = df.withColumn(col, funct(df[col]))
            types = df.select(col).distinct().collect()
            types = [ty[col] for ty in types]
            types_list = [when(df[col] == ty, 1).otherwise(0).alias('d_' + col + '_' + ty) for ty in types if
                          ty not in [None, '0', 0]]
            df = df.select(list(df.columns) + types_list)
            variable_dummy += ['d_' + col + '_' + ty for ty in types if ty not in [None, '0', 0]]
            df = df.drop(col)

        # Agrupamos las dummies por siniestro
        variable_dummy += ['po_pago_indicador_indem', 'po_pago_rehusado', 'po_pago_gasto_codigo_detective',
                           'persona_objeto_asegurado']

        for col in variable_dummy:
            name = col + '_count'
            df = df.fillna({col: 0})
            df = df.withColumn(col, df[col].cast(IntegerType()))
            df = df.withColumn(name, sum_(df[col]).over(w))
            df = df.drop(col)

        # Ajustamos las situaciones para representarlo porcentualmente:
        for col in list(df.columns):
            if col.startswith('d_po_res_situacion'):
                df = df.withColumn(col, df[col] / df['po_reserva_pagoxsiniestro_count'])

        # Ajustamos po_pago_indicador_indem para verlo porcentualmente
        df = df.withColumn('po_pago_indicador_indem_count', df['po_pago_indicador_indem_count'] / df[
            'po_reserva_pagoxsiniestro_count'])

        # FLOAT VARIABLES
        # Agrupamos las FLOAT por siniestro y las ponemos para analizar como Outliers
        variable_float_perceptor = ['po_pago_factura_importe_neto', 'po_pago_indemnizacion_importe_neto',
                                    'po_gasto_factura_importe_neto', 'po_gasto_indemnizacion_importe_neto',
                                    'po_pago_importe_neto', 'po_gasto_importe_neto',
                                    'po_pago_importe_neto_ASEGURADO',
                                    'po_pago_importe_porcentual_ASEGURADO',
                                    ]
        float_outliers = []
        for col in variable_float_perceptor:
            name = col + '_count'
            df = df.fillna({col: 0})
            df = df.withColumn(col, df[col].cast(FloatType()))
            df = df.withColumn(name, sum_(df[col]).over(w))
            float_outliers.append(name)

        # Porcentual_Asegurado: Lo ajustamos para que refleje el valor porcentual
        df = df.withColumn('po_pago_importe_porcentual_ASEGURADO_count',
                           df['po_pago_importe_porcentual_ASEGURADO_count'] / df['po_reserva_indemxsiniestro_count'])

        # TABLA DE PERCEPTOR Y SERVICIOS
        df = df.withColumn('po_pago_perceptor',
                           when(df['po_pago_perceptor'].startswith('AIDE ASISTENCIA'), 'AIDE ASISTENCIA').otherwise(
                               df['po_pago_perceptor']))

        if self._is_diario:
            df = df.join(aux_perceptor, on='po_pago_perceptor', how='left')
            df = df.join(aux_servicios.drop('id_siniestro'), on='po_gasto_perceptor', how='left')
            for col in list(aux_perceptor.columns) + list(aux_servicios.columns):
                if col not in ['po_pago_perceptor', 'po_gasto_perceptor']:
                    df = df.fillna({col: 0})
            del aux_perceptor
            del aux_servicios

        else:
            # TABLA DE PERCEPTOR
            df.withColumn('po_pago_perceptor',
                          when(df['po_pago_perceptor'].isin([0, '0']), None).otherwise(df['po_pago_perceptor']))
            w_perceptor = (Window().partitionBy('po_pago_perceptor').rowsBetween(-sys.maxsize, sys.maxsize))

            # Por Pagos
            df = df.withColumn('pondera_perceptor', when(df['po_pago_perceptor'].isNotNull(), 1).otherwise(0))
            df = df.withColumn('po_pagos_total_countxperceptor', sum_(df['pondera_perceptor']).over(w_perceptor))
            df = df.drop('pondera_perceptor')

            # Pago anulación
            df = df.fillna({'po_pago_es_anulacion': 0})
            df = df.withColumn('po_pago_es_anulacion_countxperceptor',
                               sum_(df['po_pago_es_anulacion'].cast(IntegerType())).over(w_perceptor))
            df = df.drop('po_pago_es_anulacion')

            variable_float_perceptor = ['po_pago_factura_importe_neto', 'po_pago_indemnizacion_importe_neto']
            # Nota: está bien que haya valores negativos porque son los recobros a otras empresas
            for col in variable_float_perceptor:
                name = col + '_countxperceptor'
                df = df.withColumn(name, sum_(df[col].cast(FloatType())).over(w_perceptor))
                df = df.drop(col)

            # Count a nivel siniestro
            df = df.withColumn('po_pagoxsiniestro_countxperceptor',
                               size(collect_set(df['id_siniestro']).over(w_perceptor)))

            # Obtenemos los niveles promedio por Perceptor-Siniestro
            for col in ['po_pago_factura_importe_neto_countxperceptor',
                        'po_pago_indemnizacion_importe_neto_countxperceptor']:
                df = df.withColumn(col + 'xpromediosiniestro', df[col] / df['po_pagoxsiniestro_countxperceptor'])

            # Perceptor Aparece en blacklist y cuántas
            # FUE UN SINIESTRO FRAUDULENTO? We check if the id_siniestro is associated with a previous Fraud Sinister
            bl_processed = bl_processed.select('id_siniestro').dropDuplicates(subset=['id_siniestro'])
            bl_processed = bl_processed.withColumn('po_reserva_perceptor_fraud', lit(1))
            df = df.join(bl_processed, on='id_siniestro', how='left')
            df = df.withColumn('po_reserva_perceptor_fraud',
                               when(df['po_reserva_perceptor_fraud'].isNull(), 0).otherwise(
                                   df['po_reserva_perceptor_fraud']))

            perc_aux = df.select(['po_pago_perceptor', 'id_siniestro', 'po_reserva_perceptor_fraud']).dropDuplicates(
                subset=['po_pago_perceptor', 'id_siniestro'])
            perc_aux = perc_aux.groupBy('po_pago_perceptor').agg(
                sum_('po_reserva_perceptor_fraud').alias('po_fraude_countxperceptor'))
            df = df.join(perc_aux, on='po_pago_perceptor', how='left')
            del perc_aux

            df = df.withColumn('po_fraude_porcentaje_perceptor',
                               df['po_fraude_countxperceptor'] / df['po_pagoxsiniestro_countxperceptor'])

            df.select(['po_pago_perceptor', 'po_pagos_total_countxperceptor', 'po_pago_es_anulacion_countxperceptor',
                       'po_pago_factura_importe_neto_countxperceptor',
                       'po_pago_indemnizacion_importe_neto_countxperceptor', 'po_pagoxsiniestro_countxperceptor',
                       'po_pago_factura_importe_neto_countxperceptorxpromediosiniestro',
                       'po_pago_indemnizacion_importe_neto_countxperceptorxpromediosiniestro',
                       'po_fraude_countxperceptor', 'po_fraude_porcentaje_perceptor']
                      ).dropDuplicates(subset=['po_pago_perceptor']).coalesce(1).write.mode("overwrite").option(
                "header", "true").option("sep", ";").csv(STRING.training_auxiliar_perceptor)

            # TABLA DE SERVICIOS
            df = df.withColumn('po_gasto_perceptor',
                               when(df['po_gasto_perceptor'].isin([0, '0']), None).otherwise(df['po_gasto_perceptor']))
            w_servicio = (Window().partitionBy('po_gasto_perceptor').rowsBetween(-sys.maxsize, sys.maxsize))

            # Por Pagos
            df = df.withColumn('pondera_servicio', when(df['po_gasto_perceptor'].isNotNull(), 1).otherwise(0))
            df = df.withColumn('po_pagos_total_countxservicios', sum_(df['pondera_servicio']).over(w_servicio))
            df = df.drop('pondera_servicio')

            variable_float_servicio = ['po_gasto_factura_importe_neto', 'po_gasto_indemnizacion_importe_neto']
            # Nota: está bien que haya valores negativos porque son los recobros a otras empresas
            for col in variable_float_servicio:
                name = col + '_countxservicios'
                df = df.withColumn(name, sum_(df[col].cast(FloatType())).over(w_servicio))
                df = df.drop(col)

            # Count a nivel siniestro
            df = df.withColumn('po_pagoxsiniestro_countxservicios',
                               size(collect_set(df['id_siniestro']).over(w_servicio)))

            # Obtenemos el promedio global por Servicio
            for col in ['po_gasto_factura_importe_neto_countxservicios',
                        'po_gasto_indemnizacion_importe_neto_countxservicios']:
                df = df.withColumn(col + 'xpromediosiniestro', df[col] / df['po_pagoxsiniestro_countxservicios'])

            # Perceptor Aparece en blacklist y cuántas
            # FUE UN SINIESTRO FRAUDULENTO? We check if the id_siniestro is associated with a previous Fraud Sinister
            bl_processed = bl_processed.withColumn('po_reserva_servicios_fraud', lit(1))
            df = df.join(bl_processed[['id_siniestro', 'po_reserva_servicios_fraud']], on='id_siniestro', how='left')
            df = df.withColumn('po_reserva_servicios_fraud',
                               when(df['po_reserva_servicios_fraud'].isNull(), 0).otherwise(
                                   df['po_reserva_servicios_fraud']))
            del bl_processed

            serv_aux = df.select(['po_gasto_perceptor', 'id_siniestro', 'po_reserva_servicios_fraud']).dropDuplicates(
                subset=['po_gasto_perceptor', 'id_siniestro'])
            serv_aux = serv_aux.groupBy('po_gasto_perceptor').agg(
                sum_('po_reserva_servicios_fraud').alias('po_fraude_countxservicios'))
            df = df.join(serv_aux, on='po_gasto_perceptor', how='left')
            del serv_aux

            df = df.withColumn('po_fraude_porcentaje_servicios',
                               df['po_fraude_countxservicios'] / df['po_pagoxsiniestro_countxservicios'])

            df.select(['po_gasto_perceptor', 'id_siniestro', 'po_pagos_total_countxservicios',
                       'po_gasto_factura_importe_neto_countxservicios',
                       'po_gasto_indemnizacion_importe_neto_countxservicios', 'po_pagoxsiniestro_countxservicios',
                       'po_gasto_factura_importe_neto_countxserviciosxpromediosiniestro',
                       'po_gasto_indemnizacion_importe_neto_countxserviciosxpromediosiniestro',
                       'po_fraude_countxservicios', 'po_fraude_porcentaje_servicios']
                      ).dropDuplicates(subset=['po_gasto_perceptor']).coalesce(1).write.mode("overwrite").option(
                "header", "true").option("sep", ";").csv(STRING.training_auxiliar_servicios)


        df = df.drop(*['po_pago_perceptor', 'po_gasto_perceptor'])
        df = df.dropDuplicates(subset=['id_siniestro'])
        df = df.fillna({'po_fraude_countxservicios': 0, 'po_fraude_porcentaje_servicios': 0})

        if self._is_diario:
            df_processed = df_processed.select(int_outliers + float_outliers)
            for col in int_outliers + float_outliers:
                df = outliers.Outliers.outliers_test_values(df, df_processed, col, not_count_zero=True)
        else:
            for col in int_outliers + float_outliers:
                df = outliers.Outliers.outliers_mad(df, col, not_count_zero=True)

        # DELETE VARIABLES
        del_variables = ['id_poliza', 'version_poliza', "po_res_garantia_id",
                         "po_res_cobertura_id", 'po_res_limite',
                         "po_pago_IBAN", "po_pago_emision", "po_pago_factura_fecha_emision",
                         'po_pago_factura_importe_neto', 'po_pago_indemnizacion_importe_neto',
                         'po_gasto_IBAN', 'po_gasto_emision', 'po_gasto_factura_fecha_emision',
                         'po_pago_es_anulacion', 'po_gasto_es_anulacion', 'pondera_siniestro',
                         'po_gasto_perceptor', 'po_gasto_factura_importe_neto', 'po_gasto_indemnizacion_importe_neto',
                         'po_pago_importe_neto', 'po_gasto_importe_neto',
                         'po_pago_importe_neto_ASEGURADO', 'po_pago_importe_porcentual_ASEGURADO',
                         'audit_poliza_producto_tecnico',
                         'audit_siniestro_codigo_compania', 'po_reserva_indemxsiniestro_count'
                         ]

        df = df.drop(*del_variables)

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.poreservable_output_prediction
        else:
            name = STRING.poreservable_output_training

        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main para test
if __name__ == '__main__':
    PoReserva(is_diario=False).run()
