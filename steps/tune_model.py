import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, precision_recall_curve, auc, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
from scipy.stats import uniform
from scipy.stats import randint
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import mlflow
from steps.preprocess_data import CAT_FEATURES


# for tree based model, no need to scale numerical variable and use ordinal encoder for categorical variable
categorical_transformer = Pipeline(
    steps=[("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", max_categories=10, unknown_value=-1))
           ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, CAT_FEATURES),],

    remainder='passthrough'
)


def tune_model(X_train_path, y_train_path, n_trials, class_balance, objective_metric, undersample):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    def objective(param):
        mlflow.set_experiment("loan")
        experiment = mlflow.get_experiment_by_name("loan")
        mlflow.set_tag("model", "HistGradientBoosting")
        # client = mlflow.tracking.MlflowClient()
        # run = client.create_run(experiment.experiment_id)
        with mlflow.start_run(nested=True):

            mlflow.log_params(param)
            XX_train, XX_test, yy_train, yy_test = train_test_split(
                X_train, y_train, stratify=y_train, test_size=0.2)

            if undersample:
                pipe = Pipeline(steps=[("undersampler", RandomUnderSampler()),
                                       ("preprocessor", preprocessor),
                                       ('classifier', HistGradientBoostingClassifier())])
            else:
                pipe = Pipeline(steps=[("preprocessor", preprocessor),
                                       ('classifier', HistGradientBoostingClassifier())])

            pipe.set_params(**param)

            pipe.fit(XX_train, yy_train.values.ravel())
            yy_pred_proba = pipe.predict_proba(XX_test)
            yy_pred = pipe.predict(XX_test)
            ap = average_precision_score(yy_test, yy_pred_proba[:, 1])
            roc = roc_auc_score(yy_test, yy_pred_proba[:, 1])
            acc = accuracy_score(yy_test, yy_pred)
            logloss = log_loss(yy_test, yy_pred_proba[:, 1])
            mlflow.log_metric("val PRC_AUC", ap)
            mlflow.log_metric("val ROC_AUC", roc)
            mlflow.log_metric("val accuracy", acc)
            mlflow.log_metric("val log_loss", logloss)

            if objective_metric == 'logloss':
                return {"loss": logloss,
                        "status": STATUS_OK}
            elif objective_metric == 'ap':
                return {"loss": -ap,
                        "status": STATUS_OK}
            elif objective_metric == 'roc':
                return {"loss": -roc,
                        "status": STATUS_OK}
            elif objective_metric == 'acc':
                return {"loss": -acc,
                        "status": STATUS_OK}
            else:
                raise ValueError(
                    f"Objective metric {objective_metric} is not supported.")

    # Define parameter space using hyperopt random variables
    if class_balance:
        param = {
            'classifier__learning_rate': hp.uniform('learning_rate', 0, 1),
            'classifier__max_depth': scope.int(hp.quniform('max_depth', 2, 10, 2)),
            'classifier__min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 5, 200)),
            'classifier__l2_regularization': hp.uniform('l2_regularization', 1, 10),
            'classifier__max_leaf_nodes': scope.int(hp.uniform('max_leaf_nodes', 5, 200)),
            'classifier__class_weight': hp.choice('class_weight', ['balanced'])
        }

    else:
        param = {
            'classifier__learning_rate': hp.uniform('learning_rate', 0, 1),
            'classifier__max_depth': scope.int(hp.quniform('max_depth', 2, 10, 2)),
            'classifier__min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 5, 200)),
            'classifier__l2_regularization': hp.uniform('l2_regularization', 1, 10),
            'classifier__max_leaf_nodes': scope.int(hp.uniform('max_leaf_nodes', 5, 200))
        }

    # Set up trials for tracking
    trials = Trials()

    # Pass objective fn and params to fmin() to get results
    with mlflow.start_run():
        result = fmin(fn=objective,
                      space=param,
                      algo=tpe.suggest,
                      trials=trials,
                      max_evals=n_trials)
        result = {k: float(v) for k, v in result.items()}
        mlflow.log_dict(result, "best_params.json")
    return param, result
