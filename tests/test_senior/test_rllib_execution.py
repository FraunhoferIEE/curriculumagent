# import os
# import random
#
# import pytest
# import ray
# from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib
# import ray.tune as tune
# from ray.tune.schedulers import PopulationBasedTraining
#
#
# class TestRllibExecution():
#     """
#     Testing whether the rllib run works
#     """
#     #@pytest.mark.slow
#     def test_run_rllib(self, test_paths_env):
#         """
#         Testing whether the rllib code works
#         """
#         cpus = os.cpu_count()
#         ray.init(num_cpus=cpus, num_gpus=0)
#
#         test_env_path, test_action_path = test_paths_env
#         env_config = {"action_space_path": test_action_path.as_posix(),
#                       "env_path": test_env_path.as_posix()}
#
#         pbt = PopulationBasedTraining(
#             time_attr="training_iteration",
#             metric="episode_reward_mean",
#             mode="max",
#             perturbation_interval=2,
#             resample_probability=0.5,
#             # Specifies the mutations of these hyperparams
#             hyperparam_mutations={
#                 "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
#                 # "sample_batch_size": lambda: random.randint(1, 128),
#                 # "train_batch_size": lambda: random.randint(2048, 8096),
#                 # "sgd_minibatch_size": lambda: random.randint(20, 128),
#                 "num_sgd_iter": lambda: random.randint(3, 10),
#                 "vf_loss_coeff": lambda: random.uniform(0.5, 1),
#                 "clip_param": lambda: random.uniform(0.01, 0.5),
#                 "gamma": lambda: random.uniform(0.975, 1),
#                 "entropy_coeff": lambda: 10 ** -random.uniform(2, 5)
#             })
#
#         tune.run(
#             "PPO",
#             checkpoint_freq=1,
#             scheduler=pbt,
#             keep_checkpoints_num=5,
#             verbose=1,
#             max_failures=3,
#             num_samples=1,
#             local_dir="~/bm",
#             stop={"training_iteration": 10},
#             config={
#                 "env": SeniorEnvRllib,
#                 "env_config": env_config,
#                 "num_workers": cpus -2 ,
#                 "num_envs_per_worker": 5,
#                 "lr": 5e-5,
#                 "num_gpus": 0.3333,
#                 "num_cpus_per_worker": 1,
#                 "remote_worker_envs": False,
#                 "model": {"use_lstm": False, "fcnet_hiddens": [1000, 1000, 1000, 1000], "fcnet_activation": 'relu',
#                           "vf_share_layers": True},
#
#             },
#         )
#         ray.shutdown()

