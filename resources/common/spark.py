import time
from abc import ABC, abstractmethod

from pyspark.sql import SparkSession

# from zanalytics_arch_logger.zaa_logger import ZaaLogger


class SparkJob(ABC):
    '''
    Interface that serves as a base for any Spark job
    The method run must be implemented by the child class
    Finally, the start method is used for starting the Job
    '''
    @abstractmethod
    def run(self): raise NotImplementedError

    @staticmethod
    def get_spark_session(app_name="PMP Batch"):
        return SparkSession.builder\
            .master("local[*]")\
            .config('spark.executor.memory', '8g')\
            .config('spark.driver.memory', '16g')\
            .config("spark.sql.shuffle.partitions", 10)\
            .appName(app_name)\
            .getOrCreate()

    # This method executes the run method from the child class
    '''
    def start(self):
        logger = ZaaLogger()
        class_name = self.__class__.__name__
        try:
            logger.info("Running {}".format(class_name))
            start = time.time()
            self.run()
            end = time.time()
            logger.info("{} done in {}s".format(class_name, end-start))
        except:
            logger.exception("Error executing run method from {} class".format(self.__class__.__name__))
            raise  
    '''