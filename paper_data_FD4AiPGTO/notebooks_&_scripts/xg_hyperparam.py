import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score 
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval 
from hyperopt.early_stop import no_progress_loss
import pickle
import pandas as pd
import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb


def objective(trial, X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, y_train, enable_categorical = True)
    dval = xgb.DMatrix(X_val, y_val, enable_categorical = True)
    #dtest = xgb.DMatrix(X_test, y_test, enable_categorical = True)

    param = {
        "verbosity": 0,
        "objective": "multi:softprob",
        "early_stopping_rounds":5,
        "num_class": 4,
        # use exact for small dataset.
        "tree_method": "hist",
        # defines booster, gblinear for linear functions.
        "booster": "gbtree",#trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-10, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-10, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        #"n_estimators": trial.suggest_int("n_estimators",100, 10000, step=50),

    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 2, 15, step=1)
        #'max_leaves': trial.suggest_int('max_leaves', 20,1025),
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dval)
    pred_labels = preds.argmax(axis=1)
    accuracy = sklearn.metrics.accuracy_score(y_val, pred_labels)
    return accuracy


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

    
    study = optuna.create_study(study_name="xg_optim", direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=500, n_jobs=10) #callbacks=[callback]

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
       