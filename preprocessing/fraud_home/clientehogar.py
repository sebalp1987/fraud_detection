from pyspark.sql.functions import when
from pyspark.sql.types import IntegerType, FloatType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home.outliers import Outliers


class ClienteHogar(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Cliente_hogar")

    def run(self):
        df, cliente_hogar_base = self._extract_data()
        df = self._transform_data(df, cliente_hogar_base)
        self._load_data(df)
        self._spark.stop()

    def _extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """
        if self._is_diario:

            cliente_hogar_base = (self._spark.
                                  read.
                                  csv(STRING.cliente_hogar_input_training, header=True, sep=',', encoding='UTF-8'))

            df = (
                self._spark
                    .read
                    .csv(STRING.cliente_hogar_input_prediction, header=True, sep=','))

        else:
            cliente_hogar_base = None

            df = (
                self._spark
                    .read
                    .csv(STRING.cliente_hogar_input_training, header=True, sep=','))

        return df, cliente_hogar_base

    def _transform_data(self, df, cliente_hogar_base):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param cliente_hogar_base: Base customer
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # Carga Siniestral
        df = df.withColumnRenamed('coste_del_siniestro_por_rol', 'cliente_hogar_carga_siniestral')
        df = df.withColumn('cliente_hogar_carga_siniestral',
                           when(df['cliente_hogar_carga_siniestral'] == '?', 0.).otherwise(
                               df['cliente_hogar_carga_siniestral']))
        df = df.withColumn('cliente_hogar_carga_siniestral', df.cliente_hogar_carga_siniestral.cast(FloatType()))
        df.registerTempTable('table')
        df = self._spark.sql(
            "SELECT *, ROUND(SUM(cliente_hogar_carga_siniestral) "
            "OVER (PARTITION BY cliente_hogar_carga_siniestral), 2) AS "
            "cliente_hogar_carga_siniestral_policy_sum FROM table ORDER BY id_poliza")
        self._spark.sql('DROP TABLE IF EXISTS table')
        df = df.drop('cliente_hogar_carga_siniestral')
        df = df.withColumnRenamed('cliente_hogar_carga_siniestral_policy_sum', 'cliente_hogar_carga_siniestral')

        # OUTLIERS
        outliers_var = ['cliente_hogar_numero_siniestros_anterior', 'cliente_hogar_carga_siniestral']

        if self._is_diario:
            cliente_hogar_base.registerTempTable('table')
            cliente_hogar_base = self._spark.sql("SELECT cliente_hogar_numero_siniestros_anterior, "
                                                 "ROUND(SUM(coste_del_siniestro_por_rol) "
                                                 "OVER (PARTITION BY coste_del_siniestro_por_rol), 2) AS "
                                                 "cliente_hogar_carga_siniestral FROM table ORDER BY id_poliza")
            self._spark.sql('DROP TABLE IF EXISTS table')
            cliente_hogar_base = cliente_hogar_base.fillna(0)
            for i in outliers_var:
                df = Outliers.outliers_test_values(df, cliente_hogar_base, i, not_count_zero=True)
        else:
            for i in outliers_var:
                df = Outliers.outliers_mad(df, i, not_count_zero=True)

        # DELETE VARIABLES: We delete not necessary variables.
        delete_var = ['id_fiscal', 'id_poliza', 'version_poliza',
                      'auditFechaAperturaSiniestroReferencia', 'audit_siniestro_producto_tecnico',
                      'audit_siniestro_entidad_legal']

        df = df.drop(*delete_var)

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.
        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.cliente_hogar_output_prediction
        else:
            name = STRING.cliente_hogar_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


if __name__ == '__main__':
    ClienteHogar(True).run()
