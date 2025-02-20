echo "= Data preparation and model fitting ="
python scripts/models/1_prepare_data.py
python scripts/models/2_fit_linear_models.py
python scripts/models/3_fit_hierarchical_model.py
python scripts/models/4_evaluate_models.py
python scripts/models/5_report_results.py

echo "= Generate panels for main figure ="
python scripts/figures/main/1_plot_linear_effects.py
python scripts/figures/main/2_plot_model_preds.py
python scripts/figures/main/3_plot_multilinear_surface.py
python scripts/figures/main/4_plot_synergy_masking.py

echo "= Generate panels for supplementary figures ="
python scripts/figures/supp/1_plot_obs_model.py
python scripts/figures/supp/2_plot_cv_predictions.py
python scripts/figures/supp/3_plot_linear_effects.py
python scripts/figures/supp/4_plot_multilinear_surfaces.py
python scripts/figures/supp/5_plot_multilinear_model.py
python scripts/figures/supp/6_plot_multilinear_masking.py