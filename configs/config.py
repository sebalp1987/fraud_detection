parameters = {
    "entity": "Z",
    "diario": True,
    "init_date_new": "2017-01-01",
    "init_date_historic": "2014-01-01",
    "init_reduce_sample_date": "2017-12-31",
    "del_reduce_var": True,
    "with_feedback": True,
    "reduce_sample": True,
    "oversample_times": None,
    "neighbors": 200,
    "n_estimators": 1000,
    "max_iter": 10001,
    "batch_size_range": range(1000, 100001, 1000),
    "n_clusters": 11,
    "max_depth": None,
    "oob_score": True,
    "base_sampling": None,
    "control_sampling": 'SMOTE',
    "bootstrap": True,
    "threshold_models": 0.511557789,
    "beta": 0.5,
    "cp": ['cliente_cp', 'hogar_cp'],
    "fecha_var": ['fecha_poliza_emision', 'fecha_poliza_efecto_natural', 'fecha_poliza_efecto_mvto',
                  'fecha_poliza_vto_movimiento', 'fecha_poliza_vto_natural', 'fecha_siniestro_ocurrencia',
                  'fecha_primera_visita_peritaje', 'fecha_ultima_visita_peritaje',
                  'hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia', 'hist_siniestro_otro_ultimo_fecha_ocurrencia'
                  ],
    "variance_threshold": 0.0,
    "n_random": 1,
    "refactor_probability": 1.0
}
