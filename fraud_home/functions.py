import unidecode
import datetime
import re

def replace_dict(x, key_values, key_in_value=True):
    """
    Reemplaza
    :param x: String
    :param key_values: Dict
    :param key_in_value
    :return:
    """
    if x is not None:
        x = unidecode.unidecode(x)
        for key, value in key_values.items():
            if key_in_value:
                if key in x.upper():
                    x = x.replace(x, value)
            else:
                if x.upper() in key:
                    x = x.replace(x, value)
    return x


def replace_dict_int(x, key_values):
    """
    Reemplaza
    :param x: Integer
    :param key_values: Dict
    :param key_in_value
    :return:
    """
    for key, value in key_values.items():
        if x == key:
            x = value
    return x


def replace_dict_starswith(x, key_values):
    """
    Reemplaza x si empieza con algún key
    :param x: String
    :param key_values: Dict
    :return:
    """
    x = unidecode.unidecode(x)
    for key, value in key_values.items():
        if x.upper().startswith(key):
            x = x.replace(x, value)

    return x


def replace_dict_contain(x, key_values):
    """
     Reemplaza x si la key está contenida en x
     :param x: String
     :param key_values: Dict
     :return:
     """
    x = unidecode.unidecode(x)
    for key, value in key_values.items():
        if key in x.upper():
            x = x.replace(x, value)

    return x


def replace_date(x):
    try:
        x = datetime.datetime.strptime(x, '%Y/%m/%d')
    except ValueError:
        x = '1900-01-01'
    return x

def replace_date_bad(x):
    try:
        x = datetime.datetime.strptime(x, '%Y/%m/%d')
    except:
        x = '1900/01/01'
        x = datetime.datetime.strptime(x, '%Y/%m/%d')
    if not isinstance(x, datetime.datetime):
        x = '1900/01/01'
        x = datetime.datetime.strptime(x, '%Y/%m/%d')
    return x


def normalize_string(x):
    x = unidecode.unidecode(x)
    x = x.upper()
    x = re.sub(r'[^\w\s]', '', x)
    x = x.replace(' ', '_')

    return x
