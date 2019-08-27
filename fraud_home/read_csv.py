import numpy as np
import pandas as pd


class ReadCsv:

    @staticmethod
    def load_csv(self: str, encoding='latin1'):
        """
        It load a file using read mode and a latin encoding.
        :return: A readed file
        """
        input_file = open(self, 'r', newline='', encoding=encoding)
        return input_file

    @staticmethod
    def processing_txt_without_header(self, separator=';', nan_val='?', header=False, change_decimal=False):
        """
        It loads a .txt file as a Dataframe and return a processed df without header
        """

        df = pd.read_csv(self, sep=separator, header=None, encoding='latin1', quotechar='"')
        if header:
            df = df[1:]

        if change_decimal:
            df = pd.read_csv(self, sep=separator, header=None, encoding='latin1', quotechar='"', decimal=',')

        df = df.replace(nan_val, np.nan)

        return df

    @staticmethod
    def load_blacklist(self):
        input_file = open(self, 'r', newline='\n', encoding = 'latin1')
        return input_file

