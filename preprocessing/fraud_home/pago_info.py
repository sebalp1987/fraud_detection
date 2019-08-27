import os

from pyspark.sql.functions import when
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql.utils import AnalysisException

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING


class PagoInfo(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Pago-Info")

    def run(self):
        df, bl_processed = self._extract_data()
        df = self._transform_data(df, bl_processed)
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
                    .csv(STRING.pagos_input_prediction, header=True, sep=','))
        else:
            df = (
                self._spark
                    .read
                    .csv(STRING.pagos_input_training, header=True, sep=','))

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
        :param bl_processed: Bl processed file.
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # PAGOS SUCESIVOS: pago_forma_curso differs from pago_forma_sucesivas
        df = df.withColumn('pago_forma_sucesiva_no_coincide',
                           when(df['pago_forma_sucesivas'] != df['pago_forma_curso'], 1).otherwise(0))

        # Transform the code to description for categorical variables
        primerer_recibo = {'IN': 'Intermediario', 'BC': 'Banco', 'CD': 'Ventanilla', 'CO': 'Negocio Aceptado',
                           'CC': 'Tarjeta Credito'}

        situacion_recibo = {'A': 'Anulado', 'E': 'Emitido', 'L': 'Liquidado', 'P': 'Pendiente',
                            'A|L': 'Anulado-Liquidado', 'L|P': 'Liquidado-Pendiente'}

        forma_pago = {'A': 'Anual', 'B': 'Bimestral', 'C': 'Cuatrimestral', 'E': 'Extraordinaria',
                      'F': 'Fraccionada', 'M': 'Mensual', 'P': 'Plan Pagos', 'S': 'Semestral',
                      'T': 'Trimestral', 'U': 'Unica', 'Z': 'Aperiodica'}

        df = df.na.replace(primerer_recibo, 1, 'pago_canal_cobro_1er_recibo')
        df = df.na.replace(situacion_recibo, 1, 'pago_situacion_recibo')
        df = df.na.replace(forma_pago, 1, 'pago_forma_curso')

        # We create categories from morosity
        df = df.withColumn('pago_morosidad',
                           when(df['pago_morosidad'].isin(['9', '10']), "Bueno").otherwise(df['pago_morosidad']))
        df = df.withColumn('pago_morosidad',
                           when(df['pago_morosidad'].isin(['6', '7', '8']), "Regular").otherwise(df['pago_morosidad']))
        df = df.withColumn('pago_morosidad',
                           when(df['pago_morosidad'].isin(['1', '2', '3', '4', '5']), "Malo").otherwise(
                               df['pago_morosidad']))
        df = df.withColumn('pago_morosidad',
                           when(df['pago_morosidad'].isin(['0']), "No_Paga").otherwise(df['pago_morosidad']))

        # DUMMY Categorical Variables
        categorical_var = ['pago_canal_cobro_1er_recibo', 'pago_situacion_recibo', 'pago_morosidad',
                           'pago_forma_curso']
        for var in categorical_var:
            df = df.fillna({var: '?'})
            types = df.select(var).distinct().collect()
            types = [value[var] for value in types]
            var_type = [when(df[var] == ty, 1).otherwise(0).alias('d_' + var + '_' + ty) for ty in types]
            cols = df.columns
            df = df.select(cols + var_type)

        # If it changes the recipiets situation
        try:
            df = df.withColumn('pago_cambio_situacion_recibo',
                               when(df['d_pago_situacion_recibo_Anulado-Liquidado'] == 1,
                                    1).otherwise(0))
        except AnalysisException:
            pass
        try:
            df = df.withColumn('pago_cambio_situacion_recibo',
                               when(df['d_pago_situacion_recibo_Liquidado-Pendiente'] == 1,
                                    1).otherwise(0))
        except AnalysisException:
            pass

        # IBAN Informado
        df = df.withColumn('pago_iban_informado',
                           when(~((df['pago_IBAN'].isNull()) | (df['pago_IBAN'] == '?')), 1).otherwise(0))

        # Check IBAN in Blacklist
        bl_processed_iban = bl_processed.filter(~((bl_processed['iban'].isNull()) | (bl_processed['iban'] == '?')))
        bl_processed_iban = bl_processed_iban.select('iban')
        bl_processed_iban = bl_processed_iban.dropDuplicates(subset=['iban'])
        df = df.join(bl_processed_iban, df.pago_IBAN == bl_processed_iban.iban, how='left')
        df = df.withColumn('cliente_iban_blacklist', when(df['iban'].isNull(), 0).otherwise(1))

        # Delete Variables
        delete_var = ['id_fiscal', 'id_poliza', 'version_poliza',
                      'auditFechaOcurrenciaSiniestroReferencia', 'pago_forma_sucesivas', 'pago_IBAN',
                      'audit_poliza_producto_tecnico',
                      'audit_poliza_entidad_legal', 'iban', 'pago_canal_cobro_1er_recibo', 'pago_situacion_recibo',
                      'pago_morosidad', 'pago_forma_curso']

        df = df.drop(*delete_var)

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.pagos_output_prediction
        else:
            name = STRING.pagos_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main para test
if __name__ == '__main__':
    PagoInfo(True).run()
