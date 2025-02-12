import os
import shutil

import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ExperimentConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import optuna
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ExperimentConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
torch.set_float32_matmul_precision("medium")


def calculate_metrics(model, X):
    results = {}
    pred = model.predict(X)
    results["accuracy"] = accuracy_score(X["target"], pred["prediction"])
    results["balanced_accuracy"] = balanced_accuracy_score(X["target"], pred["prediction"])
    results["f1_micro"] = f1_score(X["target"], pred["prediction"], average="micro")
    results["f1_macro"] = f1_score(X["target"], pred["prediction"], average="macro")

    # binary
    y_binary = X["target"].copy()
    pred_binary = pred["prediction"].copy()
    y_binary[y_binary > 0] = 1
    pred_binary[pred_binary > 0] = 1

    results["binary_accuracy"] = accuracy_score(y_binary, pred_binary)
    results["binary_balanced_accuracy"] = balanced_accuracy_score(y_binary, pred_binary)
    results["binary_f1_micro"] = f1_score(y_binary, pred_binary, average="micro")
    results["binary_f1_macro"] = f1_score(y_binary, pred_binary, average="macro")

    return results

if __name__ == "__main__":
    ### PLEASE SPECIFY YOUR DATA PATH ###
    data_path = Path()

    print("Load data: ")

    train = np.load(data_path / "full_obs_data_train.npz")
    val = np.load(data_path / "full_obs_data_val.npz")
    test = np.load(data_path / "full_obs_data_test.npz")

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

    dn_test = test["dn"]
    senior_test = test["senior"]
    topo_test = test["topo"]

    dn_test = np.hstack([np.zeros((dn_test.shape[0], 1)), dn_test])
    senior_test = np.hstack([np.ones((senior_test.shape[0], 1)), senior_test])
    topo_test = np.hstack([2 * np.ones((topo_test.shape[0], 1)), topo_test])

    test = np.vstack([dn_test, senior_test, topo_test])

    del dn_test, senior_test, topo_test

    dataset_colnames = np.load(data_path / "dataset_colnames.npy")
    dataset_colnames = np.append(["agent_id"], dataset_colnames)
    dataset_colnames = np.append(dataset_colnames, "target")
    print(dataset_colnames.shape)

    X_train, y_train = train[:, :-1], train[:, -1]
    X_val, y_val = val[:, :-1], val[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    # Combine features and labels for the training set
    train = np.column_stack((X_train, y_train))

    # Combine features and labels for the validation set
    val = np.column_stack((X_val, y_val))

    # Combine features and labels for the test set
    test = np.column_stack((X_test, y_test))

    train = pd.DataFrame(train, columns=dataset_colnames)
    val = pd.DataFrame(val, columns=dataset_colnames)
    test = pd.DataFrame(test, columns=dataset_colnames)

    data_config = DataConfig(
        target=["target"],
        continuous_cols=dataset_colnames.tolist()[:-1],
        categorical_cols=[],
        # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented,
        num_workers=8,
        pin_memory=True
    )
    trainer_config = TrainerConfig(
        devices=1,
        accelerator="gpu",
        batch_size=1024,
        max_epochs=100,
        early_stopping="valid_loss",
        early_stopping_mode="min",
        early_stopping_patience=10,
        checkpoints="valid_loss",
        load_best=True,
        progress_bar="none",
        trainer_kwargs=dict(enable_model_summary=False),
    )
    optimizer_config = OptimizerConfig(optimizer="Adam",
                                       lr_scheduler="ReduceLROnPlateau",
                                       lr_scheduler_params={"mode": "min", "patience": 5, "factor": 0.5})

    head_config = LinearHeadConfig(
        layers="", dropout=0.45708582050918656, initialization="kaiming"
        # No additional layer in head, just a mapping layer to output_dim
    ).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="506-1044-574-424",
        activation="LeakyReLU",
        learning_rate=0.004892760454472693,
        head="LinearHead",
        head_config=head_config,
        dropout=0.45708582050918656
    )

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=True
    )


    print('Starting training:')

    model.fit(train=train, validation=val)

    # read ood
    dn = np.load(data_path / "OOD_DoNothing_obs_data.npy")
    senior = np.load( data_path / "OOD_Senior_original_95_obs_data.npy")
    topo = np.load(data_path / "OOD_Topo_Agent_95_2_obs_data.npy")

    dn = np.hstack([np.zeros((dn.shape[0], 1)), dn])
    # senior = np.hstack([np.zeros((senior.shape[0], 1)), senior])
    # topo = np.hstack([np.zeros((topo.shape[0], 1)), topo])
    senior = np.hstack([np.ones((senior.shape[0], 1)), senior])
    topo = np.hstack([2 * np.ones((topo.shape[0], 1)), topo])

    Xy_test_ood = np.concatenate([dn, senior, topo])
    X_test_ood = pd.DataFrame(Xy_test_ood, columns=dataset_colnames)

    train_results = calculate_metrics(model, train)
    val_results = calculate_metrics(model, val)
    test_results = calculate_metrics(model, test)
    ood_results = calculate_metrics(model, X_test_ood)
    df = pd.DataFrame([train_results, val_results, test_results, ood_results]).T
    df.columns = ["train", "val", "test", "ood"]
    print("Metrics\n")
    print(df)