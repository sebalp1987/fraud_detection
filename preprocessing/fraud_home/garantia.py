from pyspark.sql.functions import when, upper
from pyspark.sql.types import IntegerType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home.outliers import Outliers


class Garantia(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("Garantia")

    def run(self):
        df, df_base = self._extract_data()
        df = self._transform_data(df, df_base=df_base)
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
                .csv(STRING.garantia_input_prediction, header=True, sep=',', nullValue='?'))

            garantia_monthly = (
                self._spark
                .read
                .csv(STRING.garantia_input_training, header=True, sep=',', nullValue='?'))

        else:
            df = (
                self._spark
                .read
                .csv(STRING.garantia_input_training, header=True, sep=',', nullValue='?'))

            garantia_monthly = None

        return df, garantia_monthly

    def _transform_data(self, df, df_base):
        """Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumnRenamed('auditCodigoSiniestroReferencia', 'id_siniestro')
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # GARANTIA CONTINENTE FORMA: We homogenize the input data
        df = df.withColumn('garantia_continente_forma',
                           when(upper(df['garantia_continente_forma']).startswith('V'), 'valor_total').otherwise(
                               df['garantia_continente_forma']))
        df = df.withColumn('garantia_continente_forma',
                           when(upper(df['garantia_continente_forma']).startswith('P'), 'primer_riesgo').otherwise(
                               df['garantia_continente_forma']))
        df = df.withColumn('garantia_continente_forma',
                           when(df['garantia_continente_forma'].isin(['valor_total', 'primer_riesgo']),
                                df['garantia_continente_forma']).otherwise(
                               'no_identificado'))

        # GARANTIA CONTINENTE TIPO: We homogenize the input data
        df = df.withColumn('garantia_continente_tipo',
                           when(upper(df['garantia_continente_tipo']).startswith('I'), 'inmueble').otherwise(
                               df['garantia_continente_tipo']))
        df = df.withColumn('garantia_continente_tipo',
                           when(upper(df['garantia_continente_tipo']).startswith('O'), 'obras_reforma').otherwise(
                               df['garantia_continente_tipo']))
        df = df.withColumn('garantia_continente_tipo',
                           when(df['garantia_continente_tipo'].isin(['inmueble', 'obras_reforma']),
                                df['garantia_continente_tipo']).otherwise(
                               'no_identificado'))

        # DUMMIES for categorical variables
        categorical_variables = ['garantia_continente_forma', 'garantia_continente_tipo']
        for col in categorical_variables:
            types = df.select(col).distinct().collect()
            types = [ty[col] for ty in types]
            categ_type = [when(df[col] == ty, 1).otherwise(0).alias('d_' + col + '_' + ty) for ty in types]
            df = df.select(list(df.columns) + categ_type)
        df = df.drop(*categorical_variables)

        # Homogenize SI/NO variables
        dummy_var = ['garantia_robo_joyas_exclusuion', 'garantia_robo_joyas_capital_base',
                     'garantia_robo_objetos_exclusion', 'garantia_robo_objetos_capital_base']

        for col in dummy_var:
            for value in ['S', 'C', 'U']:
                df = df.withColumn(col, when(df[col].startswith(value), 1).otherwise(df[col]))
            df = df.withColumn(col, when(df[col].startswith('N'), 0).otherwise(df[col]))

        # DELETE VARIABLES
        delete_var = ['id_fiscal', 'id_poliza', 'version_poliza', 'garantia_mct', 'garantia_prod_tecnico',
                      'garantia_prod_comercial', 'auditMCT', 'auditProductoTecnico', 'auditProductoComercial',
                      'audit_poliza_entidad_legal', 'auditSiniestroProductoTecnico']

        df = df.drop(*delete_var)

        # OUTLIERS
        outliers_var = ['garantia_continente_capital_aseguado', 'garantia_contenido_capital_asegurado',
                        'garantia_robo_joyas_capital', 'garantia_robo_objetos_capital',
                        'garantia_robo_objetos_otros', 'garantia_robo_cobertura_metalico_cajafuerte',
                        'garantia_robo_cobertura_metalico_otro', 'garantia_robo_cobertura_atraco_fuera',
                        'garantia_robo_cobertura_desperfecto_inmueble']

        if self._is_diario:
            for col in outliers_var:
                Outliers.outliers_test_values(df, df_base, col, not_count_zero=True)
        else:
            for col in outliers_var:
                df = Outliers.outliers_mad(df, col, not_count_zero=True)

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.garantia_output_prediction
        else:
            name = STRING.garantia_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


# Main para test
if __name__ == '__main__':
    Garantia(False).run()
