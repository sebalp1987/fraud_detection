from pyspark.sql.functions import when, col, lit
from pyspark.sql.types import IntegerType, StructType, StructField, StringType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home.outliers import Outliers

import os

class Cliente(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Cliente")

    def run(self):
        df, bl_processed, country_list, cliente_base = self.extract_data()
        df = self.transform_data(df, bl_processed, country_list, cliente_base)
        self.load_data(df)
        self._spark.stop()

    def extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """
        if self._is_diario:
            cliente_base = (self._spark.
                            read.
                            csv(STRING.cliente_input_training, header=True, sep=',', encoding='UTF-8'))

            df = (
                self._spark
                .read
                .csv(STRING.cliente_input_prediction, header=True, sep=',', encoding='UTF-8'))

        else:
            cliente_base = None

            df = (
                self._spark
                .read
                .csv(STRING.cliente_input_training, header=True, sep=',', encoding='UTF-8'))

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

        country_list = (self._spark.
                        read.
                        csv(STRING.country_list_input, sep=';', header=True, encoding='latin1'))

        return df, bl_processed, country_list, cliente_base

    def transform_data(self, df, bl_processed, country_list, cliente_base):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param bl_processed: bl file that comes from Investigation Office
        :param country_list: country list by region
        :param cliente_base: Historical customer data
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        exprs = [col(column).alias(column.replace('"', '')) for column in df.columns]
        df = df.select(*exprs)
        exprs = [col(column).alias(column.replace(' ', '')) for column in df.columns]
        df = df.select(*exprs)
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))
        df = df.withColumn('id_fiscal', df.id_fiscal.cast(StringType()))

        # Tomador Blacklist
        bl_processed_tomador = bl_processed.filter(bl_processed['cod_rol'] == 2)
        bl_processed_tomador = bl_processed_tomador.select('nif_o_intm')
        bl_processed_tomador = bl_processed_tomador.dropDuplicates(subset=['nif_o_intm'])
        df = df.join(bl_processed_tomador, df.id_fiscal == bl_processed_tomador.nif_o_intm, how='left')
        df = df.withColumn('nif_o_intm', when(df['nif_o_intm'].isNull(), 0).otherwise(1))
        df = df.withColumnRenamed('nif_o_intm', 'cliente_id_fiscal_blacklist')

        # IBAN Blacklist
        bl_processed_iban = bl_processed.filter(~((bl_processed['iban'].isNull()) | (bl_processed['iban'] == '?')))
        bl_processed_iban = bl_processed_iban.select('iban')
        bl_processed_iban = bl_processed_iban.dropDuplicates(subset=['iban'])
        df = df.join(bl_processed_iban, df.cliente_domicilio_bancario_IBAN == bl_processed_iban.iban, how='left')
        df = df.withColumn('cliente_iban_blacklist', when(df['iban'].isNull(), 0).otherwise(1))

        # Range age customer
        df = df.withColumn('cliente_e18_29', when(df['cliente_edad'].between(18, 29), 1).otherwise(0))
        df = df.withColumn('cliente_e30_39', when(df['cliente_edad'].between(30, 39), 1).otherwise(0))
        df = df.withColumn('cliente_40_49', when(df['cliente_edad'].between(40, 49), 1).otherwise(0))
        df = df.withColumn('cliente_e50_59', when(df['cliente_edad'].between(50, 59), 1).otherwise(0))
        df = df.withColumn('cliente_e60', when(df['cliente_edad'].between(60, 100), 1).otherwise(0))
        df = df.withColumn('cliente_edad_incosistente',
                           when((~df['cliente_edad'].between(18, 100)) | (df['cliente_edad'].isNull()), 1).otherwise(0))

        # Customer Antiquity: We replace bad values with NAN. We consider > 100 because bad values are
        # associated with the bad inputation 01/01/1900
        df = df.withColumn('cliente_antiguedad',
                           when(df['cliente_antiguedad'] > 100, lit(None)).otherwise(df['cliente_antiguedad']))

        # CLIENTE NACIONALIDAD: Using a List of country regions we group the nationalities. Also we create a variable if
        # the customer is Spanish or not

        df = df.withColumn('cliente_d_español', when(df['cliente_nacionalidad'] == 'ESPAÑA', 1).otherwise(0))
        df = df.join(country_list, df.cliente_nacionalidad == country_list.COUNTRY, how='left')
        df = df.withColumnRenamed('REGION', 'cliente_region')
        df = df.drop(*['COUNTRY', 'ISO'])

        # PAIS DE RESIDENCIA: We do the same as Nationality but with the Residence Country
        df = df.withColumn('cliente_d_residencia_espania',
                           when(df['cliente_pais_residencia'] == 'ESPAÑA', 1).otherwise(0))
        df = df.join(country_list, df.cliente_pais_residencia == country_list.COUNTRY, how='left')
        df = df.withColumnRenamed('REGION', 'cliente_residencia_region')
        df = df.drop(*['COUNTRY', 'ISO'])

        # DUMMY variables
        cat_variables = ['cliente_forma_contacto', 'cliente_telefono_tipo', 'cliente_region',
                         'cliente_residencia_region']

        for var in cat_variables:
            df = df.fillna({var: '?'})
            types = df.select(var).distinct().collect()
            types = [value[var] for value in types]
            var_type = [when(df[var] == ty, 1).otherwise(0).alias('d_' + var + '_' + ty) for ty in types]
            cols = list(df.columns)
            df = df.select(cols + var_type)

        # FLOAT VARIABLES
        # 1-First, we generate the proportion between Property Sinisters and Total Sinisters that the customer had.
        df = df.withColumn('cliente_siniestro_hogar_porc', df['cliente_numero_siniestros_anterior_hogar'] /
                           df['cliente_numero_siniestros_anterior'])

        df = df.withColumn('cliente_siniestro_hogar_porc',
                           when(df['cliente_siniestro_hogar_porc'].isNull(), 0).otherwise(
                               df['cliente_siniestro_hogar_porc']))

        # 2-Second, we save each float variable to apply Outliers MAD algorithm.
        float_variables = ['cliente_numero_siniestros_anterior', 'cliente_numero_siniestros_anterior_hogar'
                           ]

        outliers = list()
        outliers += float_variables

        # OUTLIERS
        if self._is_diario:
            cliente_base = cliente_base.select(*outliers)
            for i in outliers:
                df = Outliers.outliers_test_values(df, cliente_base, i, not_count_zero=True)
        else:
            for i in outliers:
                df = Outliers.outliers_mad(df, i, not_count_zero=True)

        # DELETE VARIABLES: We delete not necessary variables.
        delete_var = ['id_fiscal', 'id_poliza', 'version_poliza', 'cliente_sexo',
                      'cliente_fecha_nacimiento',
                      'cliente_edad',
                      'cliente_morosidad', 'cliente_tipo_doc', 'cliente_apellido1', 'cliente_apellido2',
                      'cliente_nombre',
                      'cliente_pais_residencia', 'cliente_tipo_via', 'cliente_nombre_via',
                      'cliente_numero_hogar', 'cliente_puerta', 'cliente_poblacion',
                      'cliente_provincia', 'cliente_telefono_pais', 'cliente_domicilio_principal',
                      'cliente_domicilio_bancario_titular', 'cliente_domicilio_bancario_IBAN',
                      'cliente_email',
                      'cliente_nacionalidad',
                      'auditFechaAperturaSiniestroReferencia', 'cliente_telefono_numero',
                      'COUNTRY', 'REGION', 'audit_poliza_producto_tecnico',
                      'audit_poliza_entidad_legal', 'iban', 'cliente_forma_contacto', 'cliente_telefono_tipo',
                      'cliente_region', 'cliente_residencia_region']

        df = df.drop(*delete_var)

        return df

    def load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.cliente_output_prediction
        else:
            name = STRING.cliente_output_training

        # (df.toPandas().to_csv(name, header=True, sep=';', index=False))
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main for testing
if __name__ == '__main__':
    Cliente(True).run()
