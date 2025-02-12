import os
import shutil
from pathlib import Path

import lightning
import numpy as np
import pathlib
import tensorflow as tf
from lightning import Trainer
from sklearn.base import BaseEstimator
import lightning as L
import torch
from torch.nn import ReLU

from curriculumagent.junior.junior_student import Junior, load_dataset
from curriculumagent.junior.junior_student_pytorch import History, Junior_PtL

import platform
# Workaround for cross-platform torch/lightning models usage

system = platform.system()
if system == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
elif system == "Linux":
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath




class TestJuniorWithTorch:
    """
    Test Suite for the Junior model and the data loader
    """

    def test_init(self, test_submission_action_space):
        """
        Testing the normal init
        """
        junior = Junior(action_space_file=test_submission_action_space, seed=42,
                        run_with_tf=False)
        assert isinstance(junior.junior_model.lightning_model, L.LightningModule)
        assert isinstance(junior.junior_model.lightning_model.activation, ReLU)


    def test_init_scaler(self, test_submission_action_space, test_scaler):
        """
        Testing the scaler
        """
        junior = Junior(action_space_file=test_submission_action_space, seed=42, run_with_tf=False)
        assert junior.scaler is None

        junior = Junior(action_space_file=test_submission_action_space, seed=42, run_with_tf=False,
                        scaler=test_scaler)
        assert isinstance(junior.scaler, BaseEstimator)

    def test_train(self, test_submission_action_space, test_junior_input_path, test_temp_save, test_junior_input):
        """
        Testing the training of the Junior
        """
        torch.manual_seed(42)
        np.random.seed(42)
        data_path, name = test_junior_input_path
        ckpt_path = test_temp_save

        _, _, s_v, _, _, _ = test_junior_input

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        junior = Junior(action_space_file=test_submission_action_space,
                        seed=42,
                        run_with_tf=False)

        out = junior.train(
            run_name="junior",
            dataset_path=data_path,
            target_model_path=ckpt_path,
            dataset_name=name,
            epochs=30,
            patience=30
        )

        assert isinstance(out, History)
        for key in out.history.keys():
            assert key in ["loss", "val_loss", "accuracy", "val_accuracy", "val_accuracy_top20"]

        assert (ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt").is_file()
        os.remove(ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt")


        trainer = Trainer()
        out = trainer.predict(junior.junior_model.lightning_model, torch.from_numpy(s_v))

        # Test if output corresponds to the num_actions:
        assert len(out[0]) == 806

        # Check if max value occurs only once in each row
        max_value_1 = out[0].max()
        max_value_2 = out[1].max()

        assert (out[0] == max_value_1).sum().item() == 1, "Max value in out[0] occurs more than once"
        assert (out[1] == max_value_2).sum().item() == 1, "Max value in out[1] occurs more than once"

        # Remove
        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)
        assert not os.listdir(ckpt_path)

    def test_predict_after_training(self, test_submission_action_space,
                                    test_junior_input_path,
                                     test_temp_save):
        """
        Testing the training of the Junior
        """
        torch.manual_seed(42)
        data_path, name = test_junior_input_path

        ckpt_path = test_temp_save

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        junior = Junior(action_space_file=test_submission_action_space, seed=42, run_with_tf=False)

        junior.train(
            run_name="junior",
            dataset_path=data_path,
            target_model_path=ckpt_path,
            dataset_name=name,
            epochs=30,
        )
        assert (ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt").is_file()
        out = junior.test(dataset_path=data_path,
                          dataset_name=name)

        assert isinstance(out, dict)
        assert isinstance(out["accuracy of top-1"], float)
        assert isinstance(out["accuracy of top-20"], float)




    def test_predict_from_checkpoint(self, test_submission_action_space,test_junior_input_path,
                                     test_junior_input, test_temp_save):
        """
        Testing the training of the Junior
        """

        torch.manual_seed(42)
        ckpt_path = test_temp_save

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)


        data_path, name = test_junior_input_path
        junior = Junior(action_space_file=test_submission_action_space, seed=42, run_with_tf=False)

        junior.train(
            run_name="junior",
            dataset_path=data_path,
            target_model_path=ckpt_path,
            dataset_name=name,
            epochs=30,
        )

        out = junior.test(checkpoint_path= ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt",
                          dataset_path=data_path,
                          dataset_name=name)
        assert isinstance(out, dict)
        # 50% ?

        assert "accuracy of top-1" in out.keys()
        assert "accuracy of top-20" in out.keys()

        shutil.rmtree(test_temp_save)
        os.mkdir(test_temp_save)
        assert not os.listdir(test_temp_save)

    def test_main_train_function_multiple_actions(self, test_submission_action_space, test_junior_input):
        """
        Running the default train function
        """
        torch.manual_seed(42)

        path_one, path_two = test_submission_action_space
        test_data_path = Path(__file__).parent.parent / "data"

        ckpt_path = test_data_path / "temporary_save"

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)
        junior = Junior(
            action_space_file=[path_one, path_two], seed=42, run_with_tf=False)

        out =junior.train(
            run_name="junior",
            dataset_path=test_data_path / "junior_experience",
            target_model_path=ckpt_path,
            dataset_name="test",
            epochs=30,
        )

        assert isinstance(out,  History)

        # Test if last model is saved:
        checkpoint = ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt"
        assert checkpoint.is_file()

        _, _, s_v, _, _, _ = test_junior_input
        trainer = Trainer()
        out = trainer.predict(junior.junior_model.lightning_model, torch.from_numpy(s_v))

        # Test if output corresponds to the num_actions:
        assert len(out[0]) == 806
        # This should be 806, given that all two action sets total 806 actions

        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)


    def test_main_train_function_single_action_path(self, test_submission_action_space, test_junior_input_path, test_temp_save, test_junior_input):
        """
        Testing the training of the Junior
        """
        torch.manual_seed(42)
        np.random.seed(42)

        path_one, _ = test_submission_action_space
        data_path, name = test_junior_input_path
        ckpt_path = test_temp_save

        _, _, s_v, _, _, _ = test_junior_input

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        junior = Junior(action_space_file=path_one,
                        seed=42,
                        run_with_tf=False)

        out = junior.train(
            run_name="junior",
            dataset_path=data_path,
            target_model_path=ckpt_path,
            dataset_name=name,
            epochs=30,
            patience=30
        )

        assert isinstance(out, History)
        for key in out.history.keys():
            assert key in ["loss", "val_loss", "accuracy", "val_accuracy", "val_accuracy_top20"]

        assert (ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt").is_file()
        os.remove(ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt")


        trainer = Trainer()
        out = trainer.predict(junior.junior_model.lightning_model, torch.from_numpy(s_v))

        # Test if output corresponds to the num_actions:
        assert len(out[0]) == 250

        # Check if max value occurs only once in each row
        max_value_1 = out[0].max()
        max_value_2 = out[1].max()

        assert (out[0] == max_value_1).sum().item() == 1, "Max value in out[0] occurs more than once"
        assert (out[1] == max_value_2).sum().item() == 1, "Max value in out[1] occurs more than once"

        # Remove
        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)
        assert not os.listdir(ckpt_path)


class TestTorchDataset:
    """
    Testing of torch dataset
    """
    def first_test(self):
        # ToDo: Clara
        pass