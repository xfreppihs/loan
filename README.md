This project builds a Loan Default Detector to predict credit card applicants at risk of default. <br>
The original dataset can be downloaded from <a href="https://www.kaggle.com/datasets/mishra5001/credit-card?select=application_data.csv">Kaggle</a>.<br>
The two Jupyter Notebooks contain all the EDA and modeling.<br>
To run MLflow-enabled HistGradientBoosting model hyperparameter tuning:
<pre>mlflow run . --experiment-name loan -P n_trials=64 -P class_balance=True -P objective_metric=ap -P undersample=False</pre>
To run streamlit-deployed app locally:
<pre>streamlit run st-app.py</pre>
Detailed deployment on GCP can be found at <a href="https://github.com/xfreppihs/streamlit-deploytment-on-GCP">another repo of mine</a>.
