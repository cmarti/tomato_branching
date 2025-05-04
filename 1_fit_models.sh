echo "= Data preparation and model fitting ="
python scripts/models/1_prepare_data.py
python scripts/models/2_fit_linear_models.py
python scripts/models/3_fit_hierarchical_model.py
python scripts/models/4_evaluate_models.py
python scripts/models/5_report_results.py