from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING


class BlProcessed(SparkJob):

    def __init__(self):
        self._spark = self.get_spark_session("BlProcessed")

    def run(self):
        df = self._extract_data()
        df = self._transform_data(df)
        self._load_data(df)
        self._spark.stop()

    def _extract_data(self):
        """Load data from Parquet file format.
        :return: Spark DataFrame.
        """
        df = (
            self._spark
            .read
            .csv(STRING.reporting_input, header=False, sep=',',
                 encoding='UTF-8'))

        return df

    @staticmethod
    def _transform_data(df):
        """Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        rename_cols = {'_c0': 'id_siniestro', '_c1': 'id_poliza', '_c2': 'id_producto', '_c3': 'fecha_apertura',
                       '_c4': 'fecha_terminado', '_c5': 'nif_o_intm', '_c6': 'nombre', '_c7': 'nif_pagador',
                       '_c8': 'nombre_pagador', '_c9': 'iban', '_c10': 'id_mediador'}

        for old_name, new_name in rename_cols.items():
            df = df.withColumnRenamed(old_name, new_name)

        # Cast claims id
        df = df.withColumn('id_siniestro', df.id_siniestro.cast(IntegerType()))

        # We save the other participants columns in a list
        others = ['id_siniestro', 'id_poliza', 'fecha_apertura', 'fecha_terminado', 'iban'] + [col for col in df.columns
                                                                                               if
                                                                                               col.startswith('_c')]
        df_others = df.select(*others)

        # We drop others from df
        df = df.select(df.columns[:11])
        df = df.drop(*['nombre', 'nif_pagador', 'nombre_pagador', 'id_producto'])

        # We add column cod_rol and rol
        df = df.withColumn('rol', lit('Tomador'))
        df = df.withColumn('cod_rol', lit(2))

        # We take intermediary separately
        intermediary = df.drop('nif_o_intm')
        intermediary = intermediary.withColumnRenamed('id_mediador', 'nif_o_intm')
        intermediary = intermediary.withColumn('rol', lit('Intermediario'))
        intermediary = intermediary.withColumn('cod_rol', lit(3))
        intermediary = intermediary.select(['id_siniestro', 'id_poliza', 'fecha_apertura', 'fecha_terminado',
                                            'nif_o_intm', 'iban', 'rol', 'cod_rol'])
        df = df.drop('id_mediador')

        # We concat the two dataframe
        df = df.union(intermediary)

        # We return with the others and rename ('cod_rol', 'rol', 'nif_o_intm')
        for col in range(11, len(df_others.columns), 3):
            df_others_i = df_others.select(
                ['id_siniestro', 'id_poliza', 'fecha_apertura', 'fecha_terminado', '_c' + str(col + 2),
                 'iban', '_c' + str(col + 1),
                 '_c' + str(col)
                 ])
            df_others_i = df_others_i.withColumnRenamed('_c' + str(col), 'cod_rol')
            df_others_i = df_others_i.withColumnRenamed('_c' + str(col + 1), 'rol')
            df_others_i = df_others_i.withColumnRenamed('_c' + str(col + 2), 'nif_o_intm')
            df_others_i = df_others_i.dropna(thresh=1, subset='nif_o_intm')

            df = df.union(df_others_i)
        df = df.dropDuplicates()

        return df

    @staticmethod
    def _load_data(df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        # df.toPandas().to_csv(STRING.reporting_output, header=True, sep=";", encoding='UTF-8', index=False)
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").option("encoding",
                                                                                                  "UTF-8").csv(
            STRING.reporting_output)

if __name__ == '__main__':
    BlProcessed().run()
