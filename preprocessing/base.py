import datetime
import sys

from pyspark.sql.functions import when, udf, regexp_replace, lit, sum as sum_, datediff, round as round_
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, DateType
from pyspark.sql.window import Window

from fraud_home.resources.common.spark import SparkJob
from fraud_home.resources.fraud_home import STRING
from fraud_home.resources.fraud_home import functions as f, outliers

class Base(SparkJob):

    def __init__(self, is_diario):
        self._is_prediction = is_diario
        self._spark = self.get_spark_session("Base")

    def run(self):
        df, bl_processed = self._extract_data()
        df = df.limit(20)
        df = self._transform_data(df, bl_processed)
        self._load_data(df)

    def _extract_data(self):
        """Load data from Parquet file format.

        :param spark: Spark session object.
        :param file: File name as input
        :return: Spark DataFrame.
        """

        df = (
            self._spark
                .read
                .csv(STRING.mediador_input_training, header=True, sep=',', nullValue='?'))

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
                        csv(STRING.reporting_input, sep=';',
                            header=True,
                            encoding='UTF-8', schema=custom_schema))

        return df, bl_processed


    def _transform_data(self, df, bl_processed):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param bl_processed
        :return: Transformed DataFrame.
        """
        # Cast key variables and rename headers
        df = df.withColumn('mediador_cod_intermediario', df.mediador_cod_intermediario.cast(IntegerType()))
        df = df.orderBy('mediador_cod_intermediario')

        return df

    def _load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """

        (df
         .toPandas()
         .to_csv(STRING.mediador_output_training, header=True, sep=';', index=False))


# Main para test
if __name__ == '__main__':
   Base(is_diario=False).run()