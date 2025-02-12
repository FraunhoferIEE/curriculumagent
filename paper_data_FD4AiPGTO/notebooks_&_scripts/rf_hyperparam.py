from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

import pickle
import numpy as np
import pandas as pd

import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_contour

def objective(trial, X_train, y_train, X_val, y_val):

    params = {
        'n_estimators': trial.suggest_int(name="n_estimators", low=100, high=1000, step=100),
        'min_samples_split': trial.suggest_int(name="min_samples_split", low=30, high=200, step=2),
        'min_samples_leaf': trial.suggest_int(name="min_samples_leaf", low=20, high=30, step=1),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20,1025),
    }

    # Perform cross validation
    cls = RandomForestClassifier(**params)
    
    #fit model
    cls.fit(X_train,y_train)
    # Compute scores
    y_pred_val = cls.predict(X_val)   

    score = accuracy_score(y_val, y_pred_val)
    return score

if __name__ == "__main__":
    # read_dataset
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


    #storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("./journal.log"),)
    study = optuna.create_study(study_name="rf_optim", direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=500, n_jobs=100)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print(f"The highest accuracy reached by this study: {(study.best_value) * 100}%.")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")
