import re

from unidecode import unidecode

symbols = '[!@#$.,-]'


def id_conversor(tipo_doc: str, numero_doc: str):
    """
    Depending on the tipo_doc type it processes using different algorithms.
    Documents types:
    
    C = CIF
    N or R = NIF
    P = Passport
    S or T = Non-identified
    E = CEE Foreing enterprise 
    X = Not CEE Foreing enterprise
    
    Note: If it is a C we apply CIF_conversor algorithm. If it is a N or R we apply NIF_conversor
    algorithm. But if it is another type, we define it as it is correct, because it depends
    on the specificaction of each country.
    
    :param tipo_doc: Which doc type is the input
    :param numero_doc: Doc Number
    :return: Return 1 if the doc type is not correct specified
    """
    if tipo_doc == 'C':
        doc_output, correct, reason = CIF_conversor(numero_doc)

    if tipo_doc == 'N' or tipo_doc == 'R':
        doc_output, correct, reason, type_id = NIF_conversor(numero_doc)

    if tipo_doc == 'P':
        correct = 0

    if tipo_doc == 'S' or tipo_doc == 'T':
        correct = 0

    if tipo_doc == 'E':
        correct = 0

    if tipo_doc == 'X':
        correct = 0

    if type(correct) != int:
        correct = 1
    return correct


def CIF_conversor(numero_doc: str):
    """
    It takes CIF doc type and try to validate using the Spanish Law configuration.
    :param numero_doc: Doc. Number
    :return: Doc output, reason, correct
    """

    # https://es.wikipedia.org/wiki/C%C3%B3digo_de_identificaci%C3%B3n_fiscal
    doc_output = ''
    correct = 0
    reason = ''

    numero_doc = str(numero_doc)
    numero_doc = unidecode(numero_doc)
    numero_doc = numero_doc.upper()
    numero_doc = re.sub(symbols, '', numero_doc)

    length = 9

    second_third_char = ['00', '01', '02', '03', '53', '54',
                         '04', '05', '06', '07', '57', '08', '58',
                         '59', '60', '61', '62', '63', '64', '65',
                         '66', '68', '09', '10', '11', '72', '12',
                         '13', '14', '56', '15', '70', '16', '17', '55', '18', '19',
                         '20', '71', '21', '22', '23', '24', '25', '26', '27',
                         '28', '78', '79', '80', '81', '82', '83', '84', '85', '86',
                         '87', '29', '92', '93', '30', '73', '31', '32',
                         '33', '74', '34', '35', '76', '36', '94', '37', '38', '75', '39',
                         '40', '41', '90', '91', '42', '43', '77', '44', '45', '46', '96',
                         '97', '98', '47', '48', '95', '49', '50', '99', '51', '52']

    # REGLAS
    if len(numero_doc) != length:
        doc_output = 'N/A'
        correct = 1
        reason = 'Bad Length'

    else:
        num0 = numero_doc[0]
        num12 = numero_doc[1] + numero_doc[2]

        num_centrales_pares = numero_doc[2] + numero_doc[4] + numero_doc[6]
        num_centrales_impar = numero_doc[1] + numero_doc[3] + numero_doc[5] + numero_doc[7]

        num_last = numero_doc[-1]

        if num12 not in second_third_char:
            doc_output = 'N/A'
            correct = 1
            reason = 'Bad Second and Third Number'

        elif (num0 in ['P', 'Q', 'S', 'W'] or num12 == '00') and type(num_last) == int:
            doc_output = 'N/A'
            correct = 1
            reason = 'CN must be letter'

        elif num0 in ['A', 'B', 'E', 'H'] and type(num_last) == str:
            doc_output = 'N/A'
            correct = 1
            reason = 'CN must be numeric'

        else:
            # Valor Numero de Control (num_last)
            suma_par = 0
            for i in num_centrales_pares:
                i = int(i)
                suma_par += i

            suma_impar = 0
            for i in num_centrales_impar:
                i = int(i)
                i = i * 2
                s = 0
                while i:
                    s += i % 10
                    i //= 10
                suma_impar += s

            suma_total = suma_par + suma_impar
            suma_total = str(suma_total)
            last_suma = 10 - int(suma_total[-1])
            last_suma = str(last_suma)
            last_suma = int(last_suma[-1])

            if int(num_last.isdigit() == True):
                if int(num_last) == last_suma:

                    doc_output = numero_doc
                    correct = 0
                    reason = ''


                else:

                    doc_output = 'N/A'
                    correct = 1
                    reason = 'CN Bad Calculated'

            else:
                num = ['P', 'Q', 'S', 'W']
                if (num0 in num) or (num12 == '00'):

                    doc_output = numero_doc
                    correct = 0
                    reason = ''

                else:
                    if num_last.isalpha():
                        doc_output = numero_doc
                        correct = 0
                        reason = ''

    return doc_output, correct, reason


def NIF_conversor(numero_doc):
    """
    It takes NIF doc type and try to validate using the Spanish Law configuration.
    :param numero_doc: Doc. Number
    :return: Doc output, reason, correct
    """

    # https: // es.wikipedia.org / wiki / N % C3 % BAmero_de_identificaci % C3 % B3n_fiscal

    doc_output = ''
    correct = 0
    reason = ''

    numero_doc = str(numero_doc)
    numero_doc = unidecode(numero_doc)
    numero_doc = numero_doc.upper()
    numero_doc = re.sub(symbols, '', numero_doc)

    num0 = numero_doc[0]
    numerical_values = numero_doc[1:-1]
    last_num = numero_doc[-1]

    if num0.isalpha() == False:
        type_id = 'ID español'
    elif num0 == 'K':
        type_id = 'Español menor de 14 años, sin DNI'
    elif num0 == 'L':
        type_id = 'Español mayor de 14 años sin NIE'
    elif num0 == 'M':
        type_id = 'Extranjero sin NIE'
    elif num0 == 'X' or num0 == 'Y' or num0 == 'Z':
        type_id = 'Extranjero con NIE'
    else:
        type_id = 'No corresponde a un NIF válido'
        doc_output = 'N/A'
        correct = 1
        reason = 'Not Valid ID'

    # REGLAS:

    if len(numero_doc) != 9:
        doc_output = 'N/A'
        correct = 1
        reason = 'Bad Length'

    elif num0 == 'K' or num0 == 'L':
        doc_output = 'N/A'
        correct = 1
        reason = 'Underage person < 18 years'

    elif num0 == 'M':
        correct = 0
        reason = ''

    elif int(numerical_values.isdigit() == False):
        doc_output = 'N/A'
        correct = 1
        reason = 'Non-numerical Values'

    elif int(numerical_values.isdigit()):
        numerical_values = int(numerical_values)
        if num0.isdigit():
            num0 = num0
        elif num0.isalpha():
            if num0 == 'X':
                num0 = 0
            elif num0 == 'Y':
                num0 = 1
            elif num0 == 'Z':
                num0 = 2

        numerical_values = str(numerical_values)
        if len(numerical_values) < 7:
            numerical_values = numerical_values.rjust(7, '0')

        num = str(num0) + numerical_values
        try:
            letter = "TRWAGMYFPDXBNJZSQVHLCKE"[int(num) % 23]
            if str(last_num) == str(letter):
                doc_output = numero_doc
                correct = 0
                reason = ''

            else:
                doc_output = 'N/A'
                correct = 1
                reason = 'CN Bad Calculated'
        except ValueError:
            doc_output = 'N/A'
            correct = 1
            reason = 'CN Bad Calculated'

    return doc_output, correct, reason, type_id
