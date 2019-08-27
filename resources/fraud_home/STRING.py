import datetime

from fraud_home.configs import config
from fraud_home.resources.common import base_paths

# root_path = os.path.dirname(os.path.abspath(__file__))
root_path = base_paths.batch_root_path + "fraud_home/"
doc_output_path = root_path + '/data_output/'
doc_input_path = root_path + '/data_input/'
auxiliar_path = root_path + '/data_extra/'
monitoring_path= root_path + '/data_monitoring/'
model_input_path = root_path + '/model_input/'
model_output_path = root_path + '/model_output/'

# PREPROCESS
# General input paths
pipeline_preprocessing_prediction_input_path = doc_input_path
pipeline_preprocessing_training_input_path = doc_input_path
pipeline_preprocessing_prediction_output_path = doc_output_path
pipeline_preprocessing_training_output_path = doc_output_path


# Auxiliary files
redes_input = auxiliar_path + "redes.csv"
country_list_input = auxiliar_path + "country_list.csv"
feedback = auxiliar_path + 'resultado.csv'
training_auxiliar_fecha = auxiliar_path + 'auxiliar_fecha.csv'
training_auxiliar_cp = auxiliar_path + 'auxiliar_cp.csv'
training_auxiliar_perceptor = auxiliar_path + 'auxiliar_po_perceptor/'
training_auxiliar_servicios = auxiliar_path + 'auxiliar_po_servicios/'
probability_base = auxiliar_path + 'base_probabilidad.csv'

# Cliente
cliente_input_prediction = pipeline_preprocessing_prediction_input_path + "cliente_new.csv"
cliente_input_training = pipeline_preprocessing_training_input_path + "cliente_mo.csv"
cliente_output_prediction = pipeline_preprocessing_prediction_output_path + "cliente_preprocessed_prediction/"
cliente_output_training = pipeline_preprocessing_training_output_path + "cliente_preprocessed_training/"

# Cliente Hogar
cliente_hogar_input_prediction = pipeline_preprocessing_prediction_input_path + "clientehogar_new.csv"
cliente_hogar_input_training = pipeline_preprocessing_training_input_path + "clientehogar_mo.csv"
cliente_hogar_output_prediction = pipeline_preprocessing_prediction_output_path\
                                  + "cliente_hogar_preprocessed_prediction/"
cliente_hogar_output_training = pipeline_preprocessing_training_output_path + "cliente_hogar_preprocessed_training/"

# Fecha
fecha_input_prediction = pipeline_preprocessing_prediction_input_path + "fecha_new.csv"
fecha_input_training = pipeline_preprocessing_training_input_path + "fecha_mo.csv"
fecha_output_prediction = pipeline_preprocessing_prediction_output_path + "fecha_preprocessed_prediction/"
fecha_output_training = pipeline_preprocessing_training_output_path + "fecha_preprocessed_training/"

# Garantia
garantia_input_prediction = pipeline_preprocessing_prediction_input_path + "garantia_new.csv"
garantia_input_training = pipeline_preprocessing_training_input_path + "garantia_mo.csv"
garantia_output_prediction = pipeline_preprocessing_prediction_output_path + "garantia_preprocessed_prediction/"
garantia_output_training = pipeline_preprocessing_training_output_path + "garantia_preprocessed_training/"

# Historico Movimiento Póliza Otras
histmovpolotras_input_prediction = pipeline_preprocessing_prediction_input_path + "histmovpolotras_new.csv"
histmovpolotras_input_training = pipeline_preprocessing_training_input_path + "histmovpolotras_mo.csv"
histmovpolotras_output_prediction = pipeline_preprocessing_prediction_output_path + "histmovpolotras_preprocessed_prediction/"
histmovpolotras_output_training = pipeline_preprocessing_training_output_path + "histmovpolotras_preprocessed_training/"

# Historico Movimiento Póliza Referencia
histmovpolref_input_prediction = pipeline_preprocessing_prediction_input_path + "histmovpolref_new.csv"
histmovpolref_input_training = pipeline_preprocessing_training_input_path + "histmovpolref_mo.csv"
histmovpolref_output_prediction = pipeline_preprocessing_prediction_output_path + "histmovpolref_preprocessed_prediction/"
histmovpolref_output_training = pipeline_preprocessing_training_output_path + "histmovpolref_preprocessed_training/"

# Historico Siniestro Anteriores Otras Pólizas
histsinantotras_input_prediction = pipeline_preprocessing_prediction_input_path + "histsinantotras_new.csv"
histsinantotras_input_training = pipeline_preprocessing_training_input_path + "histsinantotras_mo.csv"
histsinantotras_output_prediction = pipeline_preprocessing_prediction_output_path + "histsinantotras_preprocessed_prediction/"
histsinantotras_output_training = pipeline_preprocessing_training_output_path + "histsinantotras_preprocessed_training/"

# Historico Siniestro Anteriores Referencia
histsinantref_input_prediction = pipeline_preprocessing_prediction_input_path + "histsinantref_new.csv"
histsinantref_input_training = pipeline_preprocessing_training_input_path + "histsinantref_mo.csv"
histsinantref_output_prediction = pipeline_preprocessing_prediction_output_path + "histsinantref_preprocessed_prediction/"
histsinantref_output_training = pipeline_preprocessing_training_output_path + "histsinantref_preprocessed_training/"

# Historico Siniestro Referencia
histsinref_input_prediction = pipeline_preprocessing_prediction_input_path + "histsinref_new.csv"
histsinref_input_training = pipeline_preprocessing_training_input_path + "histsinref_mo.csv"
histsinref_output_prediction = pipeline_preprocessing_prediction_output_path + "histsinref_preprocessed_prediction/"
histsinref_output_training = pipeline_preprocessing_training_output_path + "histsinref_preprocessed_training/"


# Hogar
hogar_input_prediction = pipeline_preprocessing_prediction_input_path + "hogar_new.csv"
hogar_input_training = pipeline_preprocessing_training_input_path + "hogar_mo.csv"
hogar_output_prediction = pipeline_preprocessing_prediction_output_path + "hogar_preprocessed_prediction/"
hogar_output_training = pipeline_preprocessing_training_output_path + "hogar_preprocessed_training/"

# Id
id_input_prediction = pipeline_preprocessing_prediction_input_path + "id_new.csv"
id_input_training = pipeline_preprocessing_training_input_path + "id_mo.csv"
id_output_prediction = pipeline_preprocessing_prediction_output_path + "id_preprocessed_prediction/"
id_output_training = pipeline_preprocessing_training_output_path + "id_preprocessed_training/"

# Mediador
mediador_input_training = pipeline_preprocessing_training_input_path  + "mediador_mo.csv"
mediador_output_training = pipeline_preprocessing_training_output_path + "mediador_processed/"

# Pagos
pagos_input_prediction = pipeline_preprocessing_prediction_input_path + "pagos_new.csv"
pagos_input_training = pipeline_preprocessing_training_input_path + "pagos_mo.csv"
pagos_output_prediction = pipeline_preprocessing_prediction_output_path + "pagos_preprocessed_prediction/"
pagos_output_training = pipeline_preprocessing_training_output_path + "pagos_preprocessed_training/"

# Poliza
poliza_input_prediction = pipeline_preprocessing_prediction_input_path + "poliza_new.csv"
poliza_input_training = pipeline_preprocessing_training_input_path + "poliza_mo.csv"
poliza_output_prediction = pipeline_preprocessing_prediction_output_path + "poliza_preprocessed_prediction/"
poliza_output_training = pipeline_preprocessing_training_output_path + "poliza_preprocessed_training/"

# PO Reservable
poreservable_input_prediction = pipeline_preprocessing_prediction_input_path + "poreservable_new.csv"
poreservable_input_training = pipeline_preprocessing_training_input_path + "poreservable_mo.csv"
poreservable_output_prediction = pipeline_preprocessing_prediction_output_path\
                                 + "poreservable_preprocessed_prediction/"
poreservable_output_training = pipeline_preprocessing_training_output_path + "poreservable_preprocessed_training/"
preprocessing_merge_output_prediction = pipeline_preprocessing_prediction_output_path + "merge_preprocessed_prediction.csv"
preprocessing_merge_output_training = pipeline_preprocessing_training_output_path + "merge_preprocessed_training.csv"

# Siniestro
siniestro_input_prediction = pipeline_preprocessing_prediction_input_path + "siniestro_new.csv"
siniestro_input_training = pipeline_preprocessing_training_input_path + "siniestro_mo.csv"
siniestro_output_prediction = pipeline_preprocessing_prediction_output_path + "siniestro_preprocessed_prediction/"
siniestro_output_training = pipeline_preprocessing_training_output_path + "siniestro_preprocessed_training/"

# Reporting
reporting_input = doc_input_path + "output_reporting_mo.csv"
reporting_output = doc_output_path + "reporting_preprocessed/"

# output preprocesado mensual y diario
etl_mensual = model_input_path + 'etl_mensual.csv'
etl_diaria = model_input_path + 'etl_diaria.csv'
etl_output_model = model_output_path + 'etl_diaria_' + config.parameters.get("entity") + '.csv'

# Output Monitoring
DAY = datetime.datetime.today().strftime('%Y-%m-%d')
monitoring_path_day = monitoring_path + str(DAY)
monitoring_path_no_supervisado = monitoring_path_day + '/no_supervisado/'
monitoring_path_supervisado = monitoring_path_day + '/supervisado/'
monitoring_no_supervisado_kmeans = monitoring_path_no_supervisado + "fs_kmeans_mini.csv"
monitoring_no_supervisado_uns_class = monitoring_path_no_supervisado + "unsupervised_classes.csv"
monitoring_supervisado_performance = monitoring_path_supervisado + "performance_evaluation.csv"

# Output Post Processing
red_flag_path = monitoring_path_day + "/red_flags.csv"
base_probabilidad = auxiliar_path + 'base_probabilidad.csv'
probabilidad_ambos_modelos = monitoring_path_day + '/probabilidad_ambos_modelos.csv'
control_evaluation = monitoring_path_day + '/control_evaluation.csv'
checklist = monitoring_path_day + '/checklist_' + str(DAY) + '.csv'
probabilidad_unnormalized = monitoring_path_day + '/prob_unnormalized_' + str(DAY) + '.csv'
probabilidad_normalized = monitoring_path_day + '/prob_normalized_' + config.parameters.get("entity") + '_' + str(DAY) + '.csv'

# Model Output
probabilidad_output = model_output_path + '/prob_normalized_' + config.parameters.get("entity") + '_' + str(DAY) + '.csv'

'--------------DUMMIES FILLNA -----------------------------------------------------------------------'
fillna_id = ['d_producto_', 'd_entidad_', 'd_tipodoc_']
fillna_cliente_hogar = []
fillna_cliente = ['cliente_forma_contacto_', 'cliente_telefono_tipo_', 'cliente_region_', 'cliente_residencia_region_']
fillna_fecha = ['d_fecha_siniestro_ocurrencia_year', 'dfecha_siniestro_ocurrencia_month_',
                'd_fecha_siniestro_ocurrencia_weekday_', 'd_fecha_poliza_efecto_natural_year_',
                'd_fecha_poliza_vto_natural_year_']
fillna_hist_mov_pol_otro = ['d_hist_poliz_ultimo_movimiento_', 'd_hist_poliza_estado_A_',
                            'd_hist_mov_poliza_otro_producto_ACCIDENTES_']
fillna_hist_mov_pol_ref = ['d_hist_movimiento_tipo_', 'd_hist_movimiento_tipo_']
fillna_hist_ant_otro = ['d_hist_sin_poliza_otro_producto_']
fillna_hist_sin_ant_ref = ['d_hist_sin_sit_']
fillna_hist_sin_ref = ['hist_siniestro_actual_oficina_']
fillna_hogar = ['d_tipo_hogar_', 'd_hogar_ubicacion_', 'd_hogar_caracter_', 'd_hogar_uso_']
fillna_pago = ['d_pago_canal_cobro_1er_recibo_', 'd_pago_situacion_recibo_', 'd_pago_morosidad_', 'd_pago_forma_curso_',
               ]
fillna_perito = []
fillna_reserva = []
fillna_poliza = ['poliza_desc_estructura', 'poliza_canal', 'poliza_duracion', 'poliza_credit_scoring',
                 'poliza_ultimo_movimiento']
fillna_siniestro = []
fillna_europa = []

fillna_vars = fillna_id + fillna_cliente_hogar + fillna_cliente + fillna_fecha + fillna_hist_mov_pol_otro + \
              fillna_hist_mov_pol_ref + fillna_hist_ant_otro + fillna_hist_sin_ant_ref + fillna_hist_sin_ref + \
              fillna_hogar + fillna_pago + fillna_perito + fillna_reserva + fillna_poliza + fillna_siniestro


class Parameters:
    correct_city_names = {'CALPE': 'CALP', 'ISIL': 'els-pallaresos', 'MARTORELLES': 'MARTORELL',
                          'SANTA GERTRUDIS': 'IBIZA',
                          'CIUDADELA': 'PALMA', 'MORELL, EL': 'TARRAGONA', 'ENCINAREJO DE CORDOB': 'CORDOBA',
                          'PORTALS NOUS': 'PALMA', 'OÃATI': 'OÑATE', 'VILADEMAT': 'GIRONA', 'VILADAMAT': 'GIRONA',
                          'BORRASA': 'BORRASSA', 'EREÑO': 'BILBAO', 'SANTA MARGARIDA': 'PALMA', 'BAQUEIRA': 'BENASQUE',
                          'SALITJA': 'GIRONA', 'VILADECNAS': 'VILADECANS', 'QUEIXANS': 'ALP',
                          'CALETA DE FUSTE': 'LAS PALMAS DE GRAN CANARIA', 'CELEIRO': 'BURELA DE CABO',
                          'QUINTES': 'GIJON', 'CASTELLON': 'valencia-valencia', 'CALPE': 'alicante-valencia',
                          'GILENA': 'malaga', 'ESPINOSA DE VILLAGONZALO S/N': 'burgos',
                          'SANTA MARGARIDA DE MONTBUI': 'sabadell',
                          "L'ALCUDIA": 'valencia-valencia', "ALCUDIA, L'": 'valencia-valencia'}

    Asegurado_Beneficiario_Perjudicado = ['770', '778', '779', '601', '715', '740', '850', '851', '852', '363']
    Profesional_Legal = ['774', '779', '780', '101', '102', '103', '104', '109', '113', '303', '723',
                         '724', '741', '743', '853', '861', '350', '351', '352', '354', '357', '801', '802', '803']
    Detective = ['105', '353', '804']
    Perito = ['106', '107', '108', '356', '360']
    Reparador = ['705']

    todos = Asegurado_Beneficiario_Perjudicado + Profesional_Legal + Detective + Perito + Reparador

    # GARANTIAS
    dict_garantias = {'RC': 'RC', 'CIVIL': 'RC', 'ROBO': 'ROBO', 'HURTO': 'ROBO', 'EXPO': 'ROBO', 'CRIST': 'CRISTALES',
                      'AGUA': 'AGUA', 'DAG': 'AGUA', 'ELECTR': 'ELECTRICIDAD', 'VV_DE': 'ELECTRICIDAD',
                      'ATM': 'ATMOSFERICO', 'INCENDIO': 'INCENDIO', 'ESTETICA': 'ESTETICA', 'JUR': 'DEF_JURIDICA',
                      'EXTENSION': 'EXTENSION', 'CONTINENTE': 'CONTINENTE', 'CONTENIDO': 'CONTENIDO',
                      'CTE': 'CONTINENTE', 'CDO': 'CONTENIDO'}

    dict_cobertura_1 = {'PORCALOR': 'CALOR', 'VALLAS': 'VALLAS', 'OTR': 'OTRO', 'RC': 'RC', 'ROBO': 'ROBO',
                        'HURTO': 'ROBO', 'ATR': 'ROBO', 'EXPO': 'ROBO', 'CRIS': 'CRISTALES', 'INUNDA': 'INUNDACION',
                        'ELECT': 'ELECTRICIDAD'}

    dict_cobertura_2 = {'CHOQUE': 'CHOQUE', 'INC': 'INCENDIO', 'AGUA': 'AGUA', 'RAYO': 'RAYO', 'VIENTO': 'VIENTO',
                        'LLUVIA': 'LLUVIA', 'PEDRISCO': 'PEDRISCO', 'JUR': 'DEF_JURIDICA', 'LLAVE': 'LLAVES',
                        'VANDAL': 'VANDALISMO', 'ALIM': 'ALIMENTOS', 'EXC.': 'CONTENIDO', 'JOY': 'JOYAS',
                        'METALICO': 'METALICO', 'VITRO': 'VITROCERAMICA', 'ACC': 'ACCIDENTE', 'VV_EXT': 'EXTERIOR',
                        'ESTET': 'ESTETICA', 'DAEST': 'ESTETICA', 'DEST': 'ESTETICA', 'RL': 'RL', 'INMB': 'INMUEBLE',
                        'MAT': 'MATERIAL', 'ALQ': 'ALQUILER', 'SAN': 'SANITARIO', 'FRAUD': 'FRAUDE_TARJETAS',
                        'CDO': 'CONTENIDO'}
