import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Union, List, Optional

import ray
import tensorflow as tf
from ray._raylet import ObjectRef
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import check_env
from sklearn.base import BaseEstimator

from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib
from curriculumagent.senior.rllib_execution.senior_model_rllib import Grid2OpCustomModelTF, Grid2OpCustomModelTorch
from curriculumagent.submission.my_agent import MyAgent
import torch

class Senior:
    """
    This class is the Senior agent. It is based on the open source framework RLlib and trains with the
    PPO. The Senior model requires a Junior model and ray to be initialized.

    Note that you can use this class a parent, if you want to change the default values of the
    underlying rllib_environment or the model

    """

    def __init__(self,
                 env_path: Union[str, Path],
                 action_space_path: Union[Path, List[Path]],
                 model_path: Union[Path, str],
                 ckpt_save_path: Optional[Union[Path, str]] = None,
                 run_with_tf: bool = True,
                 scaler: Optional[Union[ObjectRef, BaseEstimator, str]] = None,
                 custom_junior_config: Optional[dict] = None,
                 num_workers: Optional[int] = None,
                 subset: Optional[bool] = False,
                 env_kwargs: Optional[dict] = {},
                 rho_threshold: Optional[float] = 0.95):
        """
        The Senior requires multiple inputs for the initialization.

        After the init is complete, you can train the Senior or restore a checkpoint.
        Further, we enable a possible check of the underlying environment and model.

        Args:
            env_path: Path to the Grid2Op Environment. This has to be initialized within the methode.
            action_space_path: Either path to the actions or a list containing mutliple actions.
            model_path: The required model path loads the underling model for the senior and consist of
            one of either a Junior model or a Senior model.
            ckpt_save_path: Optional path, where the PPO should save its checkpoints. If not provided,
            scaler: If you want, you can pass a Scaler of Sklearn Model.
            Either a Sklearn Scaler or its ray ID, if the scaler is saved via ray.put().
            If scaler is provided, the environment will scale the observations based on scaler.transform()
            custom_junior_config: If the junior model is a model after the hyperparameter training, you
            need to pass the model configurations.
            ray will save them in the default directory ray_results.
            num_workers: You can configure the number of workers, based on your ray.init() configurations.
            If not specified, the PPO will used half of you CPU count.
            subset: Optional argument, whether the observations should be filtered when saved.
            The default version saves the observations according to obs.to_vect(), however if
            subset is set to True, then only the all observations regarding the lines, busses, generators and loads are
            selected.
            env_kwargs: Optional parameters for the Grid2Op environment that should be used when making the environment.
        """
        # ToDo optimize docstrings
        # Set default values:

        self.run_with_tf = run_with_tf
        self.ckpt_save_path = ckpt_save_path
        assert ray.is_initialized(), "Ray seems not to be initialized. Please use ray.init() prior to running" \
                                     "the Senior."

        if isinstance(scaler, (str,Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    scaler = pickle.load(fp)
            except Exception as e:
                scaler = None
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        if not rho_threshold:
            rho_threshold = 0.95

        self.env_config = {
            "action_space_path": action_space_path,
            "env_path": env_path,
            "action_threshold": rho_threshold,
            'subset': subset,
            'scaler': scaler,
            'topo': True,
            "env_kwargs": env_kwargs}

        self.model_config = None

        if self.run_with_tf:
            ModelCatalog.register_custom_model('Senior', Grid2OpCustomModelTF)
        else:
            ModelCatalog.register_custom_model('Senior', Grid2OpCustomModelTorch)

        if isinstance(custom_junior_config, (dict, str)):
            if isinstance(custom_junior_config, str):
                with open(custom_junior_config) as json_file:
                    custom_junior_config = json.load(json_file)

            self.model_config = {"model_path": model_path,
                                 "custom_config": custom_junior_config}
            self.__advanced_model = True
        else:

            self.model_config = {"model_path": model_path}
            self.__advanced_model = False

        # Testing of model and init the SeniorEnvRllib
        self.rllib_env: SeniorEnvRllib = None
        self.__test_env_and_model_config()

        # Now init PPO
        num_cpu = os.cpu_count()
        if not num_workers:
            num_workers = num_cpu // 2

        if self.run_with_tf:
            self.ppo_config = (
                PPOConfig().environment(env=SeniorEnvRllib, env_config=self.env_config)
                .rollouts(num_rollout_workers=num_workers)
                .framework("tf2")
                .training(model={"custom_model": "Senior",
                                 "custom_model_config": self.model_config})
                .evaluation(evaluation_num_workers=1))
        else:
            self.ppo_config = (
                PPOConfig().environment(env=SeniorEnvRllib, env_config=self.env_config)
                .rollouts(num_rollout_workers=num_workers)
                .framework("torch")
                .training(model={"custom_model": "Senior",
                                 "custom_model_config": self.model_config})
                .evaluation(evaluation_num_workers=1))


        self.ppo = self.ppo_config.build()

    def train(self, iterations: int = 1) -> dict:
        """ Train the Senior with the underlying PPO agent.

        Args:
            iterations: Number of Iterations for the PPO

        Returns: rllib output
        """
        out = None
        for i in range(iterations):
            out = self.ppo.train()

            # For every 5 steps, we save the current checkpoint:
            if i % 5 == 0:
                self.ppo.save(checkpoint_dir=self.ckpt_save_path)

        # Now save final checkpoint
        save_path = self.ppo.save(checkpoint_dir=self.ckpt_save_path)

        logging.info("An Algorithm checkpoint has been created inside directory: "
                     f"'{save_path}'.")

        return out

    def restore(self, path: Optional[Union[str, Path]]) -> None:
        """
        Restores the provided checkpoint. Alternatively you can also use the Algorithm.from_checkpoint()
        for this.
        Args:
            path: Path to checkpoint.

        Returns: None

        """
        self.ppo.restore(path)
        logging.info(f"Restored path: {path} ")


    def save_to_model(self, path: Optional[Union[str, Path]] = "."):
        """
        Saving the model for the final Agent. This model is saved as a TensorFlow model and
        can be loaded by the MyAgent method of the CurriculumAgent.

        Args:
            path: Path, where to save the model.

        Returns:

        """
        self.ppo.export_policy_model(path)
        logging.info(f"The MyAgent model is saved under {path}")

    def get_my_agent(self, path: Optional[Union[str, Path]] = ".") -> MyAgent:
        """
        Saves the Senior model and returns the final MyAgent model.

        Returns: MyAgent model with the respective action sets of the Senior

        """
        # First Save the PPO:
        self.save_to_model(path)

        # Load the my_agent:
        if self.run_with_tf:
            agent = MyAgent(
                action_space=self.rllib_env.single_env.action_space,
                model_path=path,
                action_space_path=self.env_config["action_space_path"],
                scaler=self.env_config["scaler"],
                best_action_threshold=self.env_config["action_threshold"],
                topo=self.env_config["topo"],
                subset=self.env_config["subset"],
                run_with_tf=True
            )
        else:
            agent = MyAgent(
                action_space=self.rllib_env.single_env.action_space,
                model_path=path,
                action_space_path=self.env_config["action_space_path"],
                scaler=self.env_config["scaler"],
                best_action_threshold=self.env_config["action_threshold"],
                topo=self.env_config["topo"],
                subset=self.env_config["subset"],
                run_with_tf=False
            )

        return agent

    def __test_env_and_model_config(self) -> None:
        """ This method tests, whether the inputs of the senior are sufficient enough and
        if everything works. This also means running the rllib method check_env()

        Returns: Nothing. The method should complete or else you have a problem.
        """
        # Create the senior_env_rllib:
        logging.info("Init of SeniorEnvRllib and testing one simple execution")
        self.rllib_env = SeniorEnvRllib(self.env_config)

        assert isinstance(self.rllib_env, SeniorEnvRllib), "The initialization of the SeniorEnvRllib failed!"
        # Run Environment:
        obs, _ = self.rllib_env.reset()
        term = False
        trunc = False
        while term is False and trunc is False:
            act = random.randrange(self.rllib_env.action_space.n)
            _, _, term, trunc, _ = self.rllib_env.step(act)
        obs, _ = self.rllib_env.reset()

        logging.info("The SeniorEnvRllib environment seems to run. Next, we check ray")
        check_env(self.rllib_env)
        logging.info("The SeniorEnvRllib check completed. ")

        # Now the model
        logging.info("Analyzing the Model configuration.")

        if self.run_with_tf:
            # First TF Model
            logging.info("First loading the TensorFlow model")
            model = tf.keras.models.load_model(self.model_config["model_path"])
            model.compile()
            logging.info("TF Model Import works")
        else:
            # First Torch Model
            logging.info("First loading the PyTorch model")
            model = Grid2OpCustomModelTorch(obs_space=self.rllib_env.observation_space,
                                           action_space=self.rllib_env.action_space,
                                           num_outputs=self.rllib_env.action_space.n,
                                           model_config={},
                                           model_path=self.model_config["model_path"],
                                           custom_config=self.model_config.get("custom_config", None),
                                           name="Junior")
            model._params_copy(self.model_config["model_path"])
            logging.info("PyTorch Import works")

        # Now the RLlib Model:
        if self.run_with_tf:
            if self.__advanced_model:
                model = Grid2OpCustomModelTF(obs_space=self.rllib_env.observation_space,
                                           action_space=self.rllib_env.action_space,
                                           num_outputs=self.rllib_env.action_space.n,
                                           model_config={},
                                           model_path=self.model_config["model_path"],
                                           custom_config=self.model_config["custom_config"],
                                           name="Junior")
            else:
                model = Grid2OpCustomModelTF(obs_space=self.rllib_env.observation_space,
                                           action_space=self.rllib_env.action_space,
                                           num_outputs=self.rllib_env.action_space.n,
                                           model_config={},
                                           model_path=self.model_config["model_path"],
                                           name="Junior")
        else:
            if self.__advanced_model:
                model = Grid2OpCustomModelTorch(obs_space=self.rllib_env.observation_space,
                                           action_space=self.rllib_env.action_space,
                                           num_outputs=self.rllib_env.action_space.n,
                                           model_config={},
                                           model_path=self.model_config["model_path"],
                                           custom_config=self.model_config["custom_config"],
                                           name="Junior")
            else:
                model = Grid2OpCustomModelTorch(obs_space=self.rllib_env.observation_space,
                                           action_space=self.rllib_env.action_space,
                                           num_outputs=self.rllib_env.action_space.n,
                                           model_config={},
                                           model_path=self.model_config["model_path"],
                                           name="Junior")

        logging.info("Init of model worked.")
        # Now Testing
        if self.run_with_tf:
            obs = {"obs": obs.reshape(1, -1)}
        else:
            obs = {"obs": torch.from_numpy(obs.reshape(1, -1))}
        assert model.forward(input_dict=obs, state=1, seq_lens=None), "Error, the model was not able to pass values!"
        logging.info("Model seems to be working")
        logging.info("All testing completed. ")
