import datetime
import sys

import pandas as pd
from pyspark.sql.functions import when, udf, lit, count as count_
from pyspark.sql.types import IntegerType, DateType
from pyspark.sql.window import Window
from fraud_home.resources.fraud_home.functions import replace_dict


def checklist5(df_reserva, df_id, df_reserva_new=None, df_id_new=None):
    """
    2 siniestros de robo con joyas del mismo asegurado
    :return: This return a Dataframe with the columns 'id_siniestro', 'checklist5_poliza', 'checklist5_nif', where
    'checklist5_' represents how many sinister (by policy/nif) belongs to JOYAS coverage
    """
    exprs = [df_id[column].alias(column.replace('"', '')) for column in df_id.columns]
    df_id = df_id.select(*exprs)
    exprs = [df_id[column].alias(column.replace(' ', '')) for column in df_id.columns]
    df_id = df_id.select(*exprs)

    df_reserva = df_reserva.select(['id_siniestro', 'id_poliza', 'po_res_cobertura'])
    df_id = df_id.select(['id_siniestro', 'id_fiscal'])
    if df_reserva_new is not None:
        df_reserva_new = df_reserva_new.select(['id_siniestro', 'id_poliza', 'po_res_cobertura'])
        df_reserva = df_reserva.union(df_reserva_new)

    df_reserva = df_reserva.dropDuplicates(subset=['id_siniestro', 'po_res_cobertura'])
    df_reserva = df_reserva.withColumn('po_res_cobertura',
                                       when(df_reserva['po_res_cobertura'].contains('JOY'), 'INCIDENCIA').otherwise(
                                           df_reserva['po_res_cobertura']))
    df_reserva = df_reserva.withColumn('po_res_cobertura',
                                       when(df_reserva['po_res_cobertura'].contains('ESPECIAL'),
                                            'INCIDENCIA').otherwise(
                                           df_reserva['po_res_cobertura']))
    df_reserva = df_reserva.filter(df_reserva['po_res_cobertura'] == 'INCIDENCIA')



    # We merge with ID by sinister
    if df_id_new is not None:
        exprs = [df_id_new[column].alias(column.replace('"', '')) for column in df_id_new.columns]
        df_id_new = df_id_new.select(*exprs)
        exprs = [df_id_new[column].alias(column.replace(' ', '')) for column in df_id_new.columns]
        df_id_new = df_id_new.select(*exprs)
        df_id_new = df_id_new.select(['id_siniestro', 'id_fiscal'])
        df_id = df_id.union(df_id_new)

    df_reserva = df_reserva.withColumn('id_siniestro', df_reserva.id_siniestro.cast(IntegerType()))
    df_id = df_id.withColumn('id_siniestro', df_id.id_siniestro.cast(IntegerType()))

    reserva_cobertura = df_reserva.join(df_id, 'id_siniestro', how='left')

    # We calculate the COUNT of JOYAS
    reserva_cobertura = reserva_cobertura.dropDuplicates(subset=['id_siniestro'])

    # Now we have the values by claim, we group by id_poliza and by nif
    w = (Window().partitionBy('id_siniestro').rowsBetween(-sys.maxsize, sys.maxsize))
    reserva_cobertura = reserva_cobertura.withColumn('checklist5_poliza',
                                                     count_(reserva_cobertura['id_poliza']).over(w))
    reserva_cobertura = reserva_cobertura.withColumn('checklist5_nif',
                                                     count_(reserva_cobertura['id_fiscal']).over(w))

    reserva_cobertura = reserva_cobertura.drop(*['id_poliza', 'id_fiscal', 'po_res_cobertura'])

    return reserva_cobertura


def checklist6b(df_fecha_new, df_fecha, df_reserva_new,
                df_reserva):
    """
     y 2 o mas reclamaciones por daños
    eléctricos en el año.
    his return a Dataframe with the columns 'id_siniestro', 'checklist6b', where
    'checklist6b' counts how many Electrical claims the person has (Temporal Space = 1 year before the
    claim occurance)
    """
    # First we reduce the universe of sinister by RESERVA = DAÑOS ELECTRICOS
    if df_reserva_new is not None:
        df_reserva_new = df_reserva_new.select(['id_siniestro', 'po_res_cobertura'])
        df_reserva_new = df_reserva_new.withColumn('TEST', lit(1))
        df_reserva_base = df_reserva.select(['id_siniestro', 'po_res_cobertura'])
        df_reserva_base = df_reserva_base.withColumn('TEST', lit(0))
        df_reserva = df_reserva_base.union(df_reserva_new)

    else:
        df_reserva = df_reserva.withColumn('TEST', lit(1))

    df_reserva = df_reserva.select(['id_siniestro', 'po_res_cobertura', 'TEST'])
    df_reserva = df_reserva.dropDuplicates(subset=['id_siniestro', 'po_res_cobertura'])
    df_reserva = df_reserva.withColumn('po_res_cobertura',
                                       when(df_reserva['po_res_cobertura'].like('%ELECT%'), 'INCIDENCIA').otherwise(
                                           df_reserva['po_res_cobertura']))
    df_reserva = df_reserva.withColumn('po_res_cobertura',
                                       when(df_reserva['po_res_cobertura'].like('%DE%'), 'INCIDENCIA').otherwise(
                                           df_reserva['po_res_cobertura']))
    df_reserva = df_reserva.withColumn('po_res_cobertura',
                                       when(df_reserva['po_res_cobertura'].like('%RAYO%'), 'INCIDENCIA').otherwise(
                                           df_reserva['po_res_cobertura']))

    df_reserva = df_reserva.filter(df_reserva['po_res_cobertura'] == 'INCIDENCIA')

    # We bring the test bottle and the Date Bottle (because we need the past sinister)
    if df_fecha_new is not None:
        df_fecha_new = df_fecha_new.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
        df_fecha_base = df_fecha.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())
        df_fecha_base = df_fecha_base.withColumn('fecha_siniestro_ocurrencia',
                                                 func(df_fecha_base['fecha_siniestro_ocurrencia']))
        df_fecha = df_fecha_base.union(df_fecha_new)

    df_fecha = df_fecha.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
    df_fecha = df_fecha.dropDuplicates(subset=['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])

    # We cross to the left of df_reserva to get only the sinister with INCIDENCIA
    df_reserva = df_reserva.withColumn('id_siniestro', df_reserva['id_siniestro'].cast(IntegerType()))
    df_fecha = df_fecha.withColumn('id_siniestro_fecha', df_fecha['id_siniestro'].cast(IntegerType()))

    df_fecha = df_fecha.drop('id_siniestro')
    df = df_reserva.join(df_fecha, df_reserva.id_siniestro == df_fecha.id_siniestro_fecha, how='left')
    df = df.drop('id_siniestro_fecha')
    df = df.withColumn('id_poliza', df['id_poliza'].cast('string'))
    df = df.dropna(subset='id_poliza')
    df = df.filter(df['TEST'] == 1)
    df = df.select('id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia')
    df = df.toPandas()
    df['checklist_6b'] = pd.Series(0, index=df.index)
    list_values = []

    for index, row in df.iterrows():
        fecha_ocurrencia = row['fecha_siniestro_ocurrencia']
        poliza = row['id_poliza']

        # Filtramos la póliza que estamos analizando
        df_i = df[df['id_poliza'] == poliza]
        df_i = df_i[df_i['fecha_siniestro_ocurrencia'] >= fecha_ocurrencia - datetime.timedelta(days=365)]
        df_i = df_i.drop_duplicates(subset=['id_siniestro'])

        count_sinister = df_i['id_siniestro'].count()
        row['checklist_6b'] = int(count_sinister)

        list_values.append([row['id_siniestro'], row['checklist_6b']])
    return list_values


def checklist_7(df_fecha_new, df_fecha, df_reserva_new, df_reserva):
    """"
    Un
    siniestro
    por
    anualidad
    repetitivo(atraco)
    """

    # Es parecido al de joyas. pero no me queda claro. Tengo que ver si tiene un atraco por año?
    if df_reserva_new is not None:
        df_reserva_new = df_reserva_new.select(['id_siniestro', 'po_res_cobertura'])
        df_reserva_new = df_reserva_new.withColumn('TEST', lit(1))
        df_reserva_base = df_reserva.select(['id_siniestro', 'po_res_cobertura'])
        df_reserva_base = df_reserva_base.withColumn('TEST', lit(0))
        df_reserva = df_reserva_base.union(df_reserva_new)

    else:
        df_reserva = df_reserva.withColumn('TEST', lit(1))

    df_reserva = df_reserva.select(['id_siniestro', 'po_res_cobertura', 'TEST'])
    df_reserva = df_reserva.dropDuplicates(subset=['id_siniestro', 'po_res_cobertura'])
    df_reserva = df_reserva.withColumn('po_res_cobertura',
                                       when(df_reserva['po_res_cobertura'].like('%ATR%'), 'INCIDENCIA').otherwise(
                                           df_reserva['po_res_cobertura']))
    df_reserva = df_reserva.withColumn('po_res_cobertura',
                                       when(df_reserva['po_res_cobertura'].like('%EXPO%'), 'INCIDENCIA').otherwise(
                                           df_reserva['po_res_cobertura']))

    df_reserva = df_reserva.filter(df_reserva['po_res_cobertura'] == 'INCIDENCIA')

    # We bring the test bottle and the Date Bottle (because we need the past sinister)
    if df_fecha_new is not None:
        df_fecha_new = df_fecha_new.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
        df_fecha_base = df_fecha.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())
        df_fecha_base = df_fecha_base.withColumn('fecha_siniestro_ocurrencia',
                                                 func(df_fecha_base['fecha_siniestro_ocurrencia']))
        df_fecha = df_fecha_base.union(df_fecha_new)

    df_fecha = df_fecha.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
    df_fecha = df_fecha.dropDuplicates(subset=['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])

    # We cross to the left of df_reserva to get only the sinister with INCIDENCIA
    df_reserva = df_reserva.withColumn('id_siniestro', df_reserva['id_siniestro'].cast(IntegerType()))
    df_fecha = df_fecha.withColumn('id_siniestro_fecha', df_fecha['id_siniestro'].cast(IntegerType()))

    df_fecha = df_fecha.drop('id_siniestro')
    df = df_reserva.join(df_fecha, df_reserva.id_siniestro == df_fecha.id_siniestro_fecha, how='left')
    df = df.drop('id_siniestro_fecha')
    df = df.withColumn('id_poliza', df['id_poliza'].cast('string'))
    df = df.dropna(subset='id_poliza')
    df = df.filter(df['TEST'] == 1)
    df = df.select('id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia')
    df = df.toPandas()
    df['checklist_7'] = pd.Series(0, index=df.index)
    list_values = []
    for index, row in df.iterrows():
        fecha_ocurrencia = row['fecha_siniestro_ocurrencia']
        poliza = row['id_poliza']

        # Filtramos la póliza que estamos analizando
        df_i = df[df['id_poliza'] == poliza]

        # Ahora filtramos un año para adelante y un año para atrás
        # df_i = df_i[df_i['fecha_siniestro_ocurrencia'] <= fecha_ocurrencia + datetime.timedelta(days=365)]
        df_i = df_i[df_i['fecha_siniestro_ocurrencia'] >= fecha_ocurrencia - datetime.timedelta(days=365)]
        df_i = df_i.drop_duplicates(subset=['id_siniestro'])

        count_sinister = df_i['id_siniestro'].count()
        row['checklist_7'] = int(count_sinister)
        list_values.append([row['id_siniestro'], row['checklist_7']])

    return list_values


def checklist_14(df_fecha_new, df_fecha, df_reserva_new, df_reserva):
    """Reiteración de 3 o + siniestros parecidos en el mismo riesgo en un plazo de tres meses"""

    if df_reserva_new is not None:
        df_reserva_new = df_reserva_new.select(['id_siniestro', 'po_res_cobertura'])
        df_reserva_new = df_reserva_new.withColumn('TEST', lit(1))
        df_reserva_base = df_reserva.select(['id_siniestro', 'po_res_cobertura'])
        df_reserva_base = df_reserva_base.withColumn('TEST', lit(0))
        df_reserva = df_reserva_base.union(df_reserva_new)

    else:
        df_reserva = df_reserva.withColumn('TEST', lit(1))

    df_reserva = df_reserva.select(['id_siniestro', 'po_res_cobertura', 'TEST'])
    df_reserva = df_reserva.dropDuplicates(subset=['id_siniestro', 'po_res_cobertura'])
    df_reserva_dict = {'PORCALOR': 'CALOR', 'VALLAS': 'VALLAS', 'OTR': 'OTRO', 'RC': 'RC', 'ROBO': 'ROBO',
                       'HURTO': 'ROBO',
                       'ATR': 'ROBO', 'EXPO': 'ROBO', 'CRIS': 'CRISTALES', 'INUNDA': 'INUNDACION',
                       'ELECT': 'ELECTRICIDAD',
                       'DE': 'ELECTRICIDAD', 'CHOQUE': 'CHOQUE', 'INC': 'INCENDIO', 'AGUA': 'AGUA', 'RAYO': 'RAYO',
                       'VIENTO': 'VIENTO', 'LLUVIA': 'LLUVIA', 'PEDRISCO': 'PEDRISCO', 'JUR': 'DEF_JURIDICA',
                       'LLAVE': 'LLAVES', 'VANDAL': 'VANDALISMO', 'ALIM': 'ALIMENTOS', 'EXC.': 'CONTENIDO',
                       'JOY': 'JOYAS', 'METALICO': 'METALICO', 'VITRO': 'VITROCERAMICA', 'ACC': 'ACCIDENTE',
                       'VV_EXT': 'CONTINENTE', 'ESTET': 'ESTETICA', 'DAEST': 'ESTETICA', 'DEST': 'ESTETICA',
                       'RL': 'RL', 'INMB': 'INMUEBLE', 'MAT': 'MATRIAL', 'ALQ': 'ALQUILER', 'SAN': 'SANITARIO',
                       'FRAUD': 'FRAUDE_TARJETAS', 'CDO': 'CONTENIDO'}

    func = udf(lambda x: replace_dict(x, key_values=df_reserva_dict, key_in_value=True), 'string')
    df_reserva = df_reserva.withColumn('po_res_cobertura', func(df_reserva['po_res_cobertura']))

    # We bring the test bottle and the Date Bottle (because we need the past sinister)
    if df_fecha_new is not None:
        df_fecha_new = df_fecha_new.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
        df_fecha_base = df_fecha.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
        func = udf(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'), DateType())
        df_fecha_base = df_fecha_base.withColumn('fecha_siniestro_ocurrencia',
                                                 func(df_fecha_base['fecha_siniestro_ocurrencia']))
        df_fecha = df_fecha_base.union(df_fecha_new)

    df_fecha = df_fecha.select(['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])
    df_fecha = df_fecha.dropDuplicates(subset=['id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia'])

    # We cross to the left of df_reserva to get only the sinister with INCIDENCIA
    df_reserva = df_reserva.withColumn('id_siniestro', df_reserva['id_siniestro'].cast(IntegerType()))
    df_fecha = df_fecha.withColumn('id_siniestro_fecha', df_fecha['id_siniestro'].cast(IntegerType()))

    df_fecha = df_fecha.drop('id_siniestro')
    df = df_reserva.join(df_fecha, df_reserva.id_siniestro == df_fecha.id_siniestro_fecha, how='left')
    df = df.drop('id_siniestro_fecha')
    df = df.withColumn('id_poliza', df['id_poliza'].cast('string'))
    df = df.dropna(subset='id_poliza')
    df = df.filter(df['TEST'] == 1)
    df = df.select('id_siniestro', 'id_poliza', 'fecha_siniestro_ocurrencia', 'po_res_cobertura')
    df = df.toPandas()
    df['checklist_14'] = pd.Series(0, index=df.index)
    list_values = []

    for index, row in df.iterrows():
        fecha_ocurrencia = row['fecha_siniestro_ocurrencia']
        poliza = row['id_poliza']
        cobertura = row['po_res_cobertura']

        # Filtramos la póliza que estamos analizando
        df_i = df[df['id_poliza'] == poliza]
        df_i = df_i[df_i['po_res_cobertura'] == cobertura]

        # Ahora filtramos un año para adelante y un año para atrás
        # df_i = df_i[df_i['fecha_siniestro_ocurrencia'] <= fecha_ocurrencia + datetime.timedelta(days=93)]
        df_i = df_i[df_i['fecha_siniestro_ocurrencia'] >= fecha_ocurrencia - datetime.timedelta(days=93)]
        df_i = df_i.drop_duplicates(subset=['id_siniestro'])

        count_sinister = df_i['id_siniestro'].count()

        row['checklist_14'] = int(count_sinister)
        list_values.append([row['id_siniestro'], row['checklist_14']])

    return list_values
