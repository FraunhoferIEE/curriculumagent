from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pickle
import numpy as np
import pandas as pd

import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_contour
import lightgbm

from lightgbm import early_stopping
from lightgbm import log_evaluation

def objective(trial, X_train, y_train, X_val, y_val):

    params = {
        'objective': "multiclass",
        'num_class': 4,
        'metric': "multi_logloss",
        'boosting_type': "gbdt",
        "verbosity": -1,
        "num_threads": 200,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-9, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-9, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

    }


    lgb_model = lightgbm.LGBMClassifier(**params)

    
    #fit model
    lgb_model.fit(X_train,y_train)
    # Compute scores
    y_pred_val = lgb_model.predict(X_val)
    
    score = accuracy_score(y_val, y_pred_val)
    return score




if __name__ == "__main__":
        
    train_data = np.load("full_obs_data_train.npz")
    val_data = np.load("full_obs_data_val.npz")
    test_data = np.load("full_obs_data_test.npz")
    dn = train_data["dn"]
    senior = train_data["senior"]
    topo = train_data["topo"]
    
    dn = np.hstack([np.zeros((dn.shape[0],1)),dn])
    senior = np.hstack([np.ones((senior.shape[0],1)),senior])
    topo = np.hstack([2 * np.ones((topo.shape[0],1)),topo])
    
    Xy_train = np.concatenate([dn,senior,topo],axis=0)
    X_train, y_train = Xy_train[:, :-1], Xy_train[:,-1]
    
    dn = val_data["dn"]
    senior = val_data["senior"]
    topo = val_data["topo"]
    
    dn = np.hstack([np.zeros((dn.shape[0],1)),dn])
    senior = np.hstack([np.ones((senior.shape[0],1)),senior])
    topo = np.hstack([2 * np.ones((topo.shape[0],1)),topo])
    
    Xy_val = np.concatenate([dn,senior,topo],axis=0)
    X_val, y_val = Xy_val[:, :-1], Xy_val[:,-1]

    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=500, n_jobs=10) #callbacks=[callback]

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
       