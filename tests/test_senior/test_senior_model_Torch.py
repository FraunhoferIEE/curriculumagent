import numpy as np
import pathlib
import pytest
import tensorflow as tf
from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn

from curriculumagent.junior.junior_student_pytorch import Junior_PtL
from curriculumagent.senior.rllib_execution.senior_model_rllib import Grid2OpCustomModelTF, Grid2OpCustomModelTorch

import platform

# Workaround for cross-platform torch/lightning models usage
system = platform.system()
if system == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
elif system == "Linux":
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath


class TestAdvancedCustomModelTorch:

    def test_load_model(self, custom_config_torch, test_submission_action_space):
        """
        First test, whether the model import works at all
        """
        model = Junior_PtL.load_from_checkpoint(checkpoint_path=custom_config_torch["model_path"],
                                                action_space_file=test_submission_action_space)
        assert isinstance(model, nn.Module)

    def test_create_model(self, obs_space, action_space, custom_config_torch):
        """
        Create Model and compare
        """
        model = Grid2OpCustomModelTorch(obs_space=obs_space, action_space=action_space,
                                        num_outputs=action_space.n, model_config={},
                                        name="test_model", **custom_config_torch)

        # Test the creation of the model
        assert model.obs_space == obs_space
        assert model.action_space == action_space
        assert model.num_outputs == action_space.n
        assert model.name == "test_model"
        assert isinstance(model.layer1, nn.Linear)
        assert isinstance(model.layer2, nn.Linear)
        assert isinstance(model.layer3, nn.Linear)
        assert isinstance(model.layer4, nn.Linear)
        assert isinstance(model.act_layer, nn.Linear)
        assert isinstance(model.val_layer, nn.Linear)

    def test_wrong_custom_config(self, obs_space, action_space, custom_config_torch):
        """
        Create Model and compare
        """

        with pytest.warns():
            cc = custom_config_torch.copy()
            cc["custom_config"] = {"what the f***": "am I doing here?"}
            Grid2OpCustomModelTorch(obs_space=obs_space, action_space=action_space,
                                    num_outputs=action_space.n, model_config={},
                                    name="test_model", **cc)

    def test_wrong_layer_shape(self, obs_space, action_space, custom_config_torch):
        """
        Testing wrong layer input
        """
        wrong_config = custom_config_torch.copy()
        wrong_config['custom_config']["layer1"] = 100
        wrong_config['custom_config']["layer2"] = 8

        with pytest.raises(RuntimeError):
            Grid2OpCustomModelTorch(obs_space=obs_space, action_space=action_space,
                                    num_outputs=action_space.n, model_config=wrong_config,
                                    name="test_model", **custom_config_torch)

    def test_wrong_shape_of_output_and_obs(self, obs_space, action_space, custom_config_torch):
        """
        Testing wrong input of action and observation
        """
        wrong_config = custom_config_torch.copy()

        wrong_obs = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        wrong_output = 2

        # Obs Space
        with pytest.raises(RuntimeError):
            Grid2OpCustomModelTorch(wrong_obs, action_space, action_space.n, wrong_config, "test_model",
                                    **custom_config_torch)

        with pytest.raises(RuntimeError):
            Grid2OpCustomModelTorch(obs_space, action_space, wrong_output, wrong_config, "test_model",
                                    **custom_config_torch)

    def test_extract_config(self, obs_space, action_space, custom_config_torch):
        """
        Create Model and test the extract_config method
        """
        # Example config
        example_config = {'activation': 'elu',
                          'layer1': 1032,
                          'layer2': 240,
                          'layer3': 1236,
                          'layer4': 1163,
                          'initializer': 'RN'}

        model = Grid2OpCustomModelTorch(obs_space, action_space, action_space.n, {}, "test_model",
                                        **custom_config_torch)

        assert example_config["initializer"] == "RN"

        layer_size, initializer, activation = model._extract_config(example_config)
        assert layer_size == [1032, 240, 1236, 1163]
        assert initializer == torch.nn.init.normal_
        assert isinstance(activation, torch.nn.ELU)

    def test_params_copy(self, obs_space, action_space, custom_config_torch):
        """
        Test the _params_copy method to verify if weights are copied correctly from the checkpoint.
        """
        # Load the checkpoint
        checkpoint = torch.load(custom_config_torch["model_path"])
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Create the model instance
        model = Grid2OpCustomModelTorch(obs_space, action_space, action_space.n, {}, "test_model",
                                        **custom_config_torch)

        # Now we reload the model with the original weights from the checkpoint
        model._params_copy(custom_config_torch["model_path"])

        # Verify that the weights were correctly copied from the checkpoint for each layer
        for layer_name, param in model.state_dict().items():
            # Add 'model.' prefix to match checkpoint's layer names
            checkpoint_layer_name = f"model.{layer_name}"

            # Check if the layer name exists in the checkpoint state_dict
            checkpoint_param = state_dict.get(checkpoint_layer_name, None)

            if checkpoint_param is not None:
                # Ensure that the weights and biases match exactly
                assert torch.allclose(param, checkpoint_param), f"Weights for layer {layer_name} do not match!"
            else:
                print(f"Layer {checkpoint_layer_name} not found in checkpoint. Skipping comparison.")

        print("All layer weights and biases match the checkpoint!")

    def test_forward(self, obs_space, action_space, custom_config_torch):
        """
        Test the forward method
        """
        model = Grid2OpCustomModelTorch(obs_space, action_space, action_space.n, {}, "test_model",
                                        **custom_config_torch)

        # Create input_dict for RLlib-style input
        input_dict = {"obs": torch.ones((1, obs_space.shape[0]))}
        action, state = model(input_dict, ["Don't mind me, just passing through"], None)

        assert action.shape == (1, action_space.n)
        assert state == ["Don't mind me, just passing through"]

        # Now we can check,whether there is a value
        value = model.value_function()
        assert isinstance(value, torch.Tensor)
        assert "_value_out" in model.__dict__

    def test_create_model_with_senior(self, obs_space, action_space, custom_config_torch, test_submission_model_torch):
        """
        Testing the import of a junior model vs. a Senior model.
        """
        model = Grid2OpCustomModelTorch(obs_space=obs_space, action_space=action_space,
                                        num_outputs=action_space.n, model_config={},
                                        name="test_model", **custom_config_torch)

        # Test the creation of the model
        weights = model.layer1.weight.data.numpy()
        assert weights.shape == (1000, obs_space.shape[0])

        # Now we try to load a Senior model
        # Note that we do not need the custom_config, because it is overwritten!
        model_path = test_submission_model_torch
        cc = {"model_path": model_path,
              "custom_config": {}}

        model = Grid2OpCustomModelTorch(obs_space=obs_space, action_space=action_space,
                                        num_outputs=action_space.n, model_config={},
                                        name="test_model", **cc)

        weights = model.layer1.weight.data.numpy()
        assert weights.shape == (1000, obs_space.shape[0])  # And not 42!
        assert weights[0][0] == pytest.approx(0.004563817)