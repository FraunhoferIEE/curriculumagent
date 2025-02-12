import os
import random
import shutil

import numpy as np
import pandas as pd
import optuna
from lightning_fabric.plugins.environments import LightningEnvironment
from pathlib import Path
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ExperimentConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig, GANDALFConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np
import pandas as pd
import optuna
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ExperimentConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

torch.set_float32_matmul_precision('medium')
import torch.distributed as dist
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
def objective_tabular(trial, X_train, y_train, X_val, y_val, dataset_colnames):
    ### PLEASE SPECIFY YOUR DATA PATH ###
    data_path = Path()

    # Define hyperparameters
    gflu_stages = trial.suggest_int("model_config__gflu_stages", 1, 10)
    dropout = trial.suggest_float("model_config__gflu_dropout", 0.01, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["RAdam", "Adam", "AdamW"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    data_config = DataConfig(
        target=[
            "target"
        ],
        continuous_cols=dataset_colnames.tolist()[:-1],
        categorical_cols=[],
        num_workers=40,
        pin_memory=True
    )

    trainer_config = TrainerConfig(
        devices=1,
        batch_size=1024,
        max_epochs=100,
        early_stopping='valid_loss',
        checkpoints_path=data_path / "saved_gandalf",
        early_stopping_patience=10,
        early_stopping_mode='min',
        load_best=True,
        trainer_kwargs=dict(enable_model_summary=False),
    )

    optimizer_config = OptimizerConfig(
        optimizer=optimizer,
        lr_scheduler="ReduceLROnPlateau",
        lr_scheduler_params={"mode": "min", "patience": 5, "factor": 0.5}
    )

    model_config = GANDALFConfig(
        task="classification",
        gflu_stages=gflu_stages,
        gflu_feature_init_sparsity=0.5,
        gflu_dropout=dropout,
        learning_rate=learning_rate,
    )

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=False
    )

    train_data = pd.DataFrame(X_train, columns=dataset_colnames[:-1])
    train_data["target"] = y_train
    val_data = pd.DataFrame(X_val, columns=dataset_colnames[:-1])
    val_data["target"] = y_val

    # Ensure the same seed is used in all processes
    torch.manual_seed(42)

    model.fit(train=train_data, validation=val_data)

    pred = model.predict(val_data)
    pred_val = pred["prediction"].copy()

    accuracy = accuracy_score(y_val, pred_val)

    for root, dirs, files in os.walk(data_path /"saved_gandalf"):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    return accuracy



if __name__ == "__main__":

    ### PLEASE SPECIFY YOUR DATA PATH ###
    data_path = Path()

    print("Load data: ")
    train = np.load(data_path / "full_obs_data_train.npz")
    val = np.load(data_path / "full_obs_data_val.npz")

    dn_train = train["dn"]
    senior_train = train["senior"]
    topo_train = train["topo"]

    dn_train = np.hstack([np.zeros((dn_train.shape[0], 1)), dn_train])
    senior_train = np.hstack([np.ones((senior_train.shape[0], 1)), senior_train])
    topo_train = np.hstack([2 * np.ones((topo_train.shape[0], 1)), topo_train])

    train = np.vstack([dn_train, senior_train, topo_train])

    del dn_train, senior_train, topo_train

    dn_val = val["dn"]
    senior_val = val["senior"]
    topo_val = val["topo"]

    dn_val = np.hstack([np.zeros((dn_val.shape[0], 1)), dn_val])
    senior_val = np.hstack([np.ones((senior_val.shape[0], 1)), senior_val])
    topo_val = np.hstack([2 * np.ones((topo_val.shape[0], 1)), topo_val])

    val = np.vstack([dn_val, senior_val, topo_val])

    del dn_val, senior_val, topo_val

    dataset_colnames = np.load(data_path / "dataset_colnames.npy")
    dataset_colnames = np.append(["agent_id"], dataset_colnames)
    dataset_colnames = np.append(dataset_colnames, "target")
    print(dataset_colnames.shape)

    X_train, y_train = train[:, :-1], train[:, -1]
    X_val, y_val = val[:, :-1], val[:, -1]

    study = optuna.create_study(direction='maximize')

    print('Starting optimization:')

    study.optimize(lambda trial: objective_tabular(trial, X_train, y_train, X_val, y_val, dataset_colnames),
                   n_trials=100, n_jobs=1, gc_after_trial=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print(f"The highest accuracy reached by this study: {(study.best_value) * 100}%.")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")