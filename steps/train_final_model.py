import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, precision_recall_curve, auc, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import mlflow
from steps.tune_model import preprocessor
from hyperopt import space_eval


def train_final_model(param, result, X_train_path, y_train_path, X_test_path, y_test_path, undersample):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    mlflow.set_experiment("loan")
    experiment = mlflow.get_experiment_by_name("loan")

    # client = mlflow.tracking.MlflowClient()
    # run = client.create_run(experiment.experiment_id)

    with mlflow.start_run():
        if undersample:
            pipe = Pipeline(steps=[("undersampler", RandomUnderSampler()),
                                   ("preprocessor", preprocessor),
                                   ('classifier', HistGradientBoostingClassifier())])
        else:
            pipe = Pipeline(steps=[("preprocessor", preprocessor),
                                   ('classifier', HistGradientBoostingClassifier())])

        pipe.set_params(**space_eval(param, result))

        pipe.fit(X_train, y_train.values.ravel())
        y_pred_proba = pipe.predict_proba(X_test)
        y_pred = pipe.predict(X_test)
        ap = average_precision_score(y_test, y_pred_proba[:, 1])
        roc = roc_auc_score(y_test, y_pred_proba[:, 1])
        acc = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba[:, 1])
        mlflow.log_metric("test PRC_AUC", ap)
        mlflow.log_metric("test ROC_AUC", roc)
        mlflow.log_metric("test accuracy", acc)
        mlflow.log_metric("test log_loss", logloss)
        mlflow.log_params(param)
        mlflow.sklearn.log_model(pipe, "histgb_model")
        mlflow.set_tag("model", "HistGradientBoosting")


if __name__ == "__main__":
    train_final_model()
