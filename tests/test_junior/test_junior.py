import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ReLU

from curriculumagent.junior.junior_student import Junior, load_dataset, train, validate


class TestJunior():
    """
    Test Suite for the Junior model and the data loader
    """

    def test_data_loading(self):
        """
        Testing the load_dataset function
        """
        data_path = Path(__file__).parent.parent / "data" / "junior_data"
        s_tr, a_tr, s_v, a_v, s_te, a_te = load_dataset(dataset_path=data_path,
                                                        dataset_name="test")
        assert s_tr.shape == (10, 1221)
        assert a_tr.shape == (10, 1)
        assert s_v.shape == (2, 1221)
        assert a_v.shape == (2, 1)
        assert s_te.shape == (1, 1221)
        assert a_te.shape == (1, 1)
        assert a_te == 11

    def test_init(self):
        """
        Testing the normal init
        """
        junior = Junior(trainset_size=1000, epochs=1,
                        num_actions=200,
                        learning_rate=1,
                        activation="relu",
                        seed=42)
        assert isinstance(junior.model, tf.keras.Model)
        assert isinstance(junior.model.layers[1].activation, ReLU)

    def test_init_leaky_relu(self):
        """
        Testin the init with the leaky relu
        """
        junior = Junior(trainset_size=1000, epochs=1,
                        num_actions=200,
                        learning_rate=1,
                        activation="leaky_relu")
        assert isinstance(junior.model, tf.keras.Model)
        assert isinstance(junior.model.layers[1].activation, LeakyReLU)

    def test_train(self, test_junior_input, test_temp_save):
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

        junior = Junior(trainset_size=1000, epochs=1,
                        num_actions=200,
                        learning_rate=1,
                        activation="leaky_relu")

        out = junior.train(log_dir=ckpt_path / "tb",
                           ckpt_dir=ckpt_path / "ckpt-junior",
                           x_train=s_tr, y_train=a_tr,
                           x_validate=s_v, y_validate=a_v,
                           epochs=30)

        assert isinstance(out,tf.keras.callbacks.History)
        for key in out.history.keys():
            key in ['loss','val_loss',"accuracy", 'val_accuracy']

        assert (ckpt_path / "tb" / "train").is_dir()
        assert (ckpt_path / "tb" / "validation").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_10").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_20").is_dir()
        assert (ckpt_path / "ckpt-junior" / "ckpt_30").is_dir()

        event = os.listdir(ckpt_path / "tb" / "train")
        assert any(["events.out.tfevents" in a for a in event])

        out = junior.model.predict(s_v)

        # Test if output corresponds to the num_actions:
        assert len(out[0]) == 200

        # Test action
        action = np.zeros(200)

        # Check if one action was selected
        assert any(out.tolist()[0])
        assert any(out.tolist()[1])

        # Remove
        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)
        assert not os.listdir(ckpt_path)

    def test_predict_after_training(self, test_junior_input,
                                    test_temp_save):
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

        junior = Junior(trainset_size=1000, epochs=1,
                        num_actions=200,
                        learning_rate=1,
                        activation="leaky_relu")

        junior.train(log_dir=ckpt_path / "tb",
                     ckpt_dir=ckpt_path / "ckpt-junior",
                     x_train=s_tr, y_train=a_tr,
                     x_validate=s_v, y_validate=a_v,
                     epochs=10)

        #
        out = junior.test(x=s_te, y=a_te)
        assert isinstance(out, dict)
        assert out['accuracy of top-1'] == 0.0
        assert out['accuracy of top-20'] == 0.0

        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)
        assert not os.listdir(ckpt_path)

    def test_predict_from_checkpoint(self, test_junior_input, test_temp_save):
        """
        Testing the training of the Junior
        """
        tf.random.set_seed(42)
        _, _, _, _, s_te, a_te = test_junior_input
        ckpt_path = test_temp_save.parent / "junior_data" / "model"

        junior = Junior(trainset_size=1000, epochs=1,
                        num_actions=30,
                        learning_rate=1e-4,
                        activation="leaky_relu")
        #
        out = junior.test(x=s_te, y=a_te, save_path=ckpt_path)
        assert isinstance(out, dict)
        assert out['accuracy of top-1'] == 0.0
        assert out['accuracy of top-20'] == 0.0

    def test_main_train_function(self, test_temp_save):
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

        train(run_name="junior",
              dataset_path=test_data_path / "junior_data",
              target_model_path=ckpt_path,
              dataset_name="test",
              epochs=30,
              seed=42)
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

    def test_main_train_function_multiple_actions(self, test_action_paths):
        """
        Running the default train function
        """
        single_path, tuple_path, tripple_path = test_action_paths
        test_data_path = Path(__file__).parent.parent / "data"

        ckpt_path = test_data_path / "temporary_save"

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        out = train(run_name="junior",
              dataset_path=test_data_path / "junior_data",
              action_space_file=[single_path, tuple_path, tripple_path],
              target_model_path=ckpt_path,
              dataset_name="test",
              epochs=30,
              seed=42)

        assert isinstance(out,tf.keras.callbacks.History)

        # Test if last model is saved:
        assert (ckpt_path / "saved_model.pb").is_file()

        # Check, wether one can load the model
        model = tf.keras.models.load_model(ckpt_path)
        assert isinstance(model, tf.keras.Model)

        out = model.output_shape
        # This should be 30, given that all three action sets total 30 actions
        assert out == (None, 30)
        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)

    def test_main_predict_function(self, test_temp_save):
        """
        Test the predict function
        """
        tf.random.set_seed(42)
        data_path = test_temp_save.parent / "junior_data"

        out = validate(checkpoint_path=data_path / "model",
                       dataset_path=data_path,
                       dataset_name="test")
        assert isinstance(out, dict)
        assert out['accuracy of top-1'] == 0.0
        assert out['accuracy of top-20'] == 0.0
