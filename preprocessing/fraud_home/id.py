from pyspark.sql.functions import when, col, udf, regexp_replace, upper
from pyspark.sql.types import IntegerType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import nif_corrector


class Id(SparkJob):

    def __init__(self, is_diario):
        self._is_diario = is_diario
        self._spark = self.get_spark_session("IdTask")

    def run(self):
        df, redes = self._extract_data()
        df = self._transform_data(df, redes, entity_='Z')
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
                .csv(STRING.id_input_prediction, header=True, sep=','))
        else:
            df = (
                self._spark
                .read
                .csv(STRING.id_input_training, header=True, sep=','))

        redes = (self._spark.
                 read.
                 csv(STRING.redes_input, header=False, sep=';'))

        return df, redes

    @staticmethod
    def _transform_data(df, redes, entity_):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param redes: Redes file that comes from Investigation Office
        :param entity_: Entity Zurich 'Z' or Another (BANC SABADELL 'BS')
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        exprs = [col(column).alias(column.replace('"', '')) for column in df.columns]
        df = df.select(*exprs)
        exprs = [col(column).alias(column.replace(' ', '')) for column in df.columns]
        df = df.select(*exprs)
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # Type of person: Fisica or Juridica
        df = df.withColumn('dummy_fisica', when(df.cliente_clase_persona_codigo == 'F', 1).otherwise(0))

        # Product TYPE dummies
        types = df.select('id_producto').distinct().collect()
        types = [i.id_producto for i in types]
        product_type = [when(df.id_producto == ty, 1).otherwise(0).alias('d_producto_' + ty) for ty in types]
        cols = list(df.columns)
        df = df.select(cols + product_type)

        # ENTITY type: Zurich or Another
        df = df.filter(df['poliza_entidad_legal'] == entity_)

        # DOC TYPE: We create dummies for National IDs types
        types = df.select('cliente_tipo_documento').distinct().collect()
        types = [i.cliente_tipo_documento for i in types]
        doc_type = [when(df.cliente_tipo_documento == ty, 1).otherwise(0).alias('d_cliente_tipo_documento_' + ty) for ty
                    in
                    types]
        cols = list(df.columns)
        df = df.select(cols + doc_type)

        # BAD ID: We check if a id is not well defined
        id_corrector = udf(lambda tipo_doc, nif: nif_corrector.id_conversor(tipo_doc, nif), IntegerType())
        df = df.withColumn('bad_id', id_corrector(df.cliente_tipo_documento, df.id_fiscal))

        # REDES: We check in our list of redes if the NIF exists
        redes = redes.withColumn('_c0', upper(regexp_replace('_c0', '-', '')))
        df = df.join(redes, df.id_fiscal == redes._c0, how='left')
        df = df.withColumn('_c0', when(df['_c0'].isNull(), 0).otherwise(1))
        df = df.withColumnRenamed('_c0', 'id_clan')

        # Drop useless columns
        df = df.drop(*['id_producto', 'id_dossier', 'poliza_entidad_legal', 'cliente_clase_persona_codigo',
                       'cliente_tipo_documento'])

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.
        :param df: DataFrame to print.
        :return: None
        """
        if self._is_diario:
            name = STRING.id_output_prediction
        else:
            name = STRING.id_output_training
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(name)


if __name__ == '__main__':
    Id(is_diario=True).run()
