import sys

import mlflow
from steps.preprocess_data import preprocess_data
from steps.tune_model import tune_model
from steps.train_final_model import train_final_model


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def pipeline():
    mlflow.set_experiment("loan")

    # step 1
    file_dirs = preprocess_data()
    print(f"{bcolors.OKCYAN}Data is preprocessed{bcolors.ENDC}")

    # step 2
    param, result = tune_model(
        X_train_path=file_dirs["X_train_dir"],
        y_train_path=file_dirs["y_train_dir"],
        n_trials=int(sys.argv[1]),
        class_balance=sys.argv[2],
        objective_metric=sys.argv[3],
        undersample=sys.argv[4]
    )
    print(f"{bcolors.OKCYAN}HP tuning is finished{bcolors.ENDC}")

    # step 3
    train_final_model(
        param,
        result,
        X_train_path=file_dirs["X_train_dir"],
        y_train_path=file_dirs["y_train_dir"],
        X_test_path=file_dirs["X_test_dir"],
        y_test_path=file_dirs["y_test_dir"],
        undersample=sys.argv[4]
    )
    print(f"{bcolors.OKGREEN}Final model is trained.{bcolors.ENDC}")


if __name__ == "__main__":
    pipeline()
