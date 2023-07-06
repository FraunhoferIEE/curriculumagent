import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import ReLU  # , LeakyRalu

from curriculumagent.junior.junior_student import Junior, load_dataset, train, validate


class TestJunior:
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
        assert s_v.shape == (2, 1429)
        assert a_v.shape == (2, 1)
        assert s_te.shape == (4, 1429)
        assert a_te.shape == (4, 1)
        assert np.array_equal(a_te.squeeze(), [6, 279, 0, 85])

    def test_init(self, test_submission_action_space):
        """
        Testing the normal init
        """
        junior = Junior(action_space_file=test_submission_action_space, seed=42)
        assert isinstance(junior.model, tf.keras.Model)
        assert isinstance(junior.model.layers[1].activation, ReLU)

    def test_init_relu(self, test_submission_action_space):
        """
        Testing the init with the leaky relu
        """
        junior = Junior(action_space_file=test_submission_action_space, seed=42)
        assert isinstance(junior.model, tf.keras.Model)
        assert isinstance(junior.model.layers[1].activation, ReLU)

    def test_train(self, test_submission_action_space, test_junior_input, test_temp_save):
        """
        Testing the training of the Junior
        """
        tf.random.set_seed(42)
        np.random.seed(42)
        s_tr, a_tr, s_v, a_v, _, _ = test_junior_input
        ckpt_path = test_temp_save

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        junior = Junior(action_space_file=test_submission_action_space, seed=42)

        out = junior.train(
            log_dir=ckpt_path / "tb",
            ckpt_dir=ckpt_path / "ckpt-junior",
            x_train=s_tr,
            y_train=a_tr,
            x_validate=s_v,
            y_validate=a_v,
            epochs=30,
        )

        assert isinstance(out, tf.keras.callbacks.History)
        for key in out.history.keys():
            key in ["loss", "val_loss", "accuracy", "val_accuracy"]

        assert (ckpt_path / "tb" / "train").is_dir()
        assert (ckpt_path / "tb" / "validation").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_10").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_20").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_30").is_dir()

        event = os.listdir(ckpt_path / "tb" / "train")
        assert "events.out.tfevents" in event[0]

        out = junior.model.predict(s_v)
        # Test if output corresponds to the num_actions:
        assert len(out[0]) == 806

        # Test action
        action = np.zeros(200)

        # Check if one action was selected
        assert any(out.tolist()[0])
        assert any(out.tolist()[1])

        # Remove
        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)
        assert not os.listdir(ckpt_path)

    def test_predict_after_training(self, test_submission_action_space,
                                    test_junior_input, test_temp_save):
        """
        Testing the training of the Junior
        """
        tf.random.set_seed(42)
        s_tr, a_tr, s_v, a_v, s_te, a_te = test_junior_input
        ckpt_path = test_temp_save

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        junior = Junior(action_space_file=test_submission_action_space, seed=42)

        junior.train(
            log_dir=ckpt_path / "tb",
            ckpt_dir=ckpt_path / "ckpt-junior",
            x_train=s_tr,
            y_train=a_tr,
            x_validate=s_v,
            y_validate=a_v,
            epochs=10,
        )

        #
        out = junior.test(x=s_te, y=a_te)
        assert isinstance(out, dict)
        assert isinstance(out["accuracy of top-1"], float)
        assert isinstance(out["accuracy of top-20"], float)

        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)
        assert not os.listdir(ckpt_path)

    def test_predict_from_checkpoint(self, test_submission_action_space,
                                     test_junior_input, test_temp_save):
        """
        Testing the training of the Junior
        """
        tf.random.set_seed(42)
        _, _, _, _, s_te, a_te = test_junior_input
        ckpt_path = test_temp_save.parent / "junior_experience" / "model"

        junior = Junior(action_space_file=test_submission_action_space, seed=42)
        #
        out = junior.test(x=s_te, y=a_te, save_path=ckpt_path)
        assert isinstance(out, dict)
        assert out["accuracy of top-1"] == 75.0
        assert out["accuracy of top-20"] == 75.0

    def test_main_train_function(self, test_submission_action_space, test_temp_save):
        """
        Running the default train function
        """
        test_data_path = Path(__file__).parent.parent / "data"

        ckpt_path = test_temp_save

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        train(
            run_name="junior",
            dataset_path=test_data_path / "junior_experience",
            action_space_file=test_submission_action_space,
            target_model_path=ckpt_path,
            dataset_name="test",
            epochs=30,
            seed=42,
        )
        # Test if last model is saved:
        assert (ckpt_path / "saved_model.pb").is_file()
        # Test if checkpoint is saved
        assert (ckpt_path / "ckpt-junior" / "ckpt_10").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_20").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_30").is_dir()

        # Check, wether one can load the model
        model = tf.keras.models.load_model(ckpt_path)
        assert isinstance(model, tf.keras.Model)

        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)

    def test_main_train_function_multiple_actions(self, test_submission_action_space):
        """
        Running the default train function
        """
        path_one, path_two = test_submission_action_space
        test_data_path = Path(__file__).parent.parent / "data"

        ckpt_path = test_data_path / "temporary_save"

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        out = train(
            run_name="junior",
            dataset_path=test_data_path / "junior_experience",
            action_space_file=[path_one, path_two],
            target_model_path=ckpt_path,
            dataset_name="test",
            epochs=30,
            seed=42,
        )

        assert isinstance(out, tf.keras.callbacks.History)

        # Test if last model is saved:
        assert (ckpt_path / "saved_model.pb").is_file()

        # Check, wether one can load the model
        model = tf.keras.models.load_model(ckpt_path)
        assert isinstance(model, tf.keras.Model)

        out = model.output_shape
        # This should be 806, given that all two action sets total 806 actions
        assert out == (None, 806)
        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)

    def test_main_predict_function(self, test_temp_save):
        """
        Test the predict function
        """
        tf.random.set_seed(42)
        data_path = test_temp_save.parent / "junior_experience"

        out = validate(checkpoint_path=data_path / "model", dataset_path=data_path, dataset_name="test")
        assert isinstance(out, dict)
        assert out["accuracy of top-1"] == 75.0
        assert out["accuracy of top-20"] == 75.0
