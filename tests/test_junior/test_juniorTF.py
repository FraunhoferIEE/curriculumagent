import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from tensorflow.keras.layers import ReLU  # , LeakyRalu

from curriculumagent.junior.junior_student import Junior, load_dataset


class TestJuniorWithTF:
    """
    Test Suite for the Junior model and the data loader
    """

    def test_data_loading(self):
        """
        Testing the load_dataset function
        """
        data_path = Path(__file__).parent.parent / "data" / "junior_experience"
        s_tr, a_tr, s_v, a_v, s_te, a_te = load_dataset(dataset_path=data_path, dataset_name="test")
        assert s_tr.shape == (3, 1429)
        assert a_tr.shape == (3, 1)
        assert s_v.shape == (4, 1429)
        assert a_v.shape == (4, 1)
        assert s_te.shape == (4, 1429)
        assert a_te.shape == (4, 1)
        assert np.array_equal(a_te.squeeze(), [6, 3, 0, 5])

    def test_init(self, test_submission_action_space):
        """
        Testing the normal init
        """
        junior = Junior(action_space_file=test_submission_action_space, seed=42, run_with_tf=True)
        assert isinstance(junior.junior_model.model, tf.keras.Model)
        assert isinstance(junior.junior_model.model.layers[1].activation, ReLU)

    def test_init_relu(self, test_submission_action_space):
        """
        Testing the init with the leaky relu
        """
        junior = Junior(action_space_file=test_submission_action_space, seed=42, run_with_tf=True)
        assert isinstance(junior.junior_model.model, tf.keras.Model)
        assert isinstance(junior.junior_model.model.layers[1].activation, ReLU)

    def test_init_scaler(self, test_submission_action_space,test_scaler):
        """
        Testing the scaler
        """
        junior = Junior(action_space_file=test_submission_action_space, seed=42, run_with_tf=True)
        assert junior.scaler is None

        junior = Junior(action_space_file=test_submission_action_space, seed=42, run_with_tf=True,
                        scaler=test_scaler)
        assert isinstance(junior.scaler,BaseEstimator)

    def test_train(self, test_submission_action_space, test_junior_input_path, test_temp_save, test_junior_input):
        """
        Testing the training of the Junior
        """
        tf.random.set_seed(42)
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
                        run_with_tf=True)

        out = junior.train(
            run_name="junior",
            dataset_path=data_path,
            target_model_path=ckpt_path,
            dataset_name=name,
            epochs=30,
            patience = 30
        )

        assert isinstance(out, tf.keras.callbacks.History)
        for key in out.history.keys():
            key in ["loss", "val_loss", "accuracy", "val_accuracy"]

        assert (ckpt_path / "ckpt-junior" / "ckpt_10").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_20").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_30").is_dir()

        out = junior.junior_model.model.predict(s_v)

        # Test if output corresponds to the num_actions:
        assert len(out[0]) == 806


        # Check if one action was selected
        assert any(out.tolist()[0])
        assert any(out.tolist()[1])

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
        tf.random.set_seed(42)
        data_path, name = test_junior_input_path

        ckpt_path = test_temp_save

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        junior = Junior(action_space_file=test_submission_action_space, seed=42)

        junior.train(
            run_name="junior",
            dataset_path=data_path,
            target_model_path=ckpt_path,
            dataset_name=name,
            epochs=30,
        )

        #
        out = junior.test(checkpoint_path=ckpt_path,
                          dataset_path=data_path,
                          dataset_name=name,)
        assert isinstance(out, dict)
        assert isinstance(out["accuracy of top-1"], float)
        assert isinstance(out["accuracy of top-20"], float)

        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)
        assert not os.listdir(ckpt_path)

    def test_predict_from_checkpoint(self, test_submission_action_space,test_junior_input_path,
                                     test_junior_input, test_temp_save):
        """
        Testing the training of the Junior
        """
        tf.random.set_seed(42)
        ckpt_path = test_temp_save.parent / "junior_experience" / "model"
        data_path, name = test_junior_input_path
        junior = Junior(action_space_file=test_submission_action_space, seed=42)
        #
        out = junior.test(checkpoint_path=ckpt_path,
                          dataset_path=data_path,
                          dataset_name=name, )
        assert isinstance(out, dict)
        # 50% ?

        assert "accuracy of top-1" in out.keys()
        assert "accuracy of top-20" in out.keys()

    def test_main_train_function_single_action_path(self, test_submission_action_space):
        """
        Running the default train function
        """
        path_one, _ = test_submission_action_space
        test_data_path = Path(__file__).parent.parent / "data"

        ckpt_path = test_data_path / "temporary_save"

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)
        junior = Junior(
            action_space_file=path_one, seed=42)

        out =junior.train(
            run_name="junior",
            dataset_path=test_data_path / "junior_experience",
            target_model_path=ckpt_path,
            dataset_name="test",
            epochs=30,
        )

        assert isinstance(out, tf.keras.callbacks.History)

        # Test if last model is saved:
        assert (ckpt_path / "saved_model.pb").is_file()

        # Check, wether one can load the model
        model = tf.keras.models.load_model(ckpt_path)
        assert isinstance(model, tf.keras.Model)

        out = model.output_shape
        # This should be 250, given used single action set has in total 250 actions
        assert out == (None, 250)
        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)