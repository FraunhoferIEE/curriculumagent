"""
This file consists of the method to train the original senior approach.

Credit: The methods are the enhanced methods of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""


import re
import time
from pathlib import Path
import tensorflow as tf
from multiprocessing import cpu_count
import numpy as np
import grid2op
from grid2op.Environment import SingleEnvMultiProcess
from curriculumagent.senior.openai_execution.ppo import PPO, PVNet
from curriculumagent.senior.openai_execution.ppo_environment import Run_env
from curriculumagent.senior.openai_execution.ppo_reward import PPO_Reward

def train_senior(run_name: str, env_path: Path, action_threshold: float = 0.9, epochs: int = 1000,
                 env_steps_per_epoch: int = 1_000):
    """ Major method to train the Senior student.

    Within this method, the junior model is loaded and the Senior model is trained based on the
    number of available cpus. The whole training process is executed on a fixed set of actions and
    with multiple Grid2Op environments in parallel.

    Note that not all GPU memory gets used from the start but it might grow a bit over time.

    Args:
        run_name: name of the experiment
        env_path: path of the underlying data for the Grid2Op environments
        action_threshold: Threshold between 0.0 and 1.0, indicating when the dangerous scenario starts.
            For example: if the threshold is set to 0.9, the agent has to act, when the max rho is larger
            or equal to 0.9
        epochs: number of epochs for training.
        env_steps_per_epoch: number of steps within the environment per epoch. Higher number of steps
            corresponds to higher computational time

    Returns: Nothing is returned but instead the results and the model is saved in the checkpoint path
        checkpoint_path = Path(f'./ckpt_{run_name}')

    """
    # Don't use all the gpu memory from the beginning but grow it
    global date
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    # hyper-parameters
    NUM_CORE = cpu_count()
    print('CPU counts：%d' % NUM_CORE)

    print(f'Run name: {run_name}')
    print(f'Hyper-parameters: cpu_core: {NUM_CORE}, env_path: {env_path}, action_threshold: {action_threshold}')

    # Build single-process environment
    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend

        backend = LightSimBackend()
        env = grid2op.make(dataset=str(env_path), backend=backend, reward_class=PPO_Reward)
    except Exception as e:
        raise e
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    # Convert to multi-process environment
    envs = SingleEnvMultiProcess(env=env, nb_env=NUM_CORE)
    envs.reset()

    # Build PPO agent
    agent = PPO(coef_entropy=1e-3, coef_value_func=0.01)

    # Build a runner
    runner = Run_env(envs, agent, num_core=NUM_CORE, action_threshold=action_threshold,
                     action_space_path='../action_space')
    # Try to load latest checkpoint if possible
    checkpoint_path = Path(f'./ckpt_{run_name}')
    print(f'Trying to load latest checkpoint from {checkpoint_path}')
    try:
        dir_pat = re.compile('([0-9]*)--?([0-9.]*)')
        checkpoint_dirs = [f for f in checkpoint_path.iterdir() if f.is_dir() and dir_pat.match(f.name)]

        # Parse directory names to figure out latest epoch checkpoint
        parsed_dirs = []
        for cd in checkpoint_dirs:
            epoch, reward = dir_pat.match(cd.name).groups()
            parsed_dirs.append((epoch, cd))
        sorted_dirs = sorted(parsed_dirs, key=lambda r: r[0])

        start_epoch, latest_checkpoint_dir = sorted_dirs[-1]
        start_epoch = int(start_epoch) + 1
        print(f"Latest checkpoint directory is: {latest_checkpoint_dir}")
        runner.agent.model.model = tf.keras.models.load_model(
            latest_checkpoint_dir, custom_objects={"PVNet": PVNet}
        )
    except Exception as e:
        print(f'Could not load checkpoint: {e}')
        start_epoch = 0

    logfile = Path(f"./log/log-{run_name}-{time.strftime('%m-%d-%H-%M', time.localtime())}")
    logfile.parent.mkdir(exist_ok=True)
    with open(logfile, 'w') as f:
        f.writelines('epoch, ave_r, ave_alive, policy_loss, value_loss, entropy, kl, clipped_ratio, time\n')

    print('start training... ...')
    for update in range(start_epoch, epochs):
        # update learning rate
        lr_now = 6e-5 * np.linspace(1, 0.025, 500)[np.clip(update, 0, 499)]
        if update < 5:
            lr_now = 1e-4
        clip_range_now = 0.2

        # generate a new batch
        tick = time.time()
        obs, returns, dones, actions, values, neg_log_p, ave_r = runner.run_n_steps(env_steps_per_epoch)
        returns /= 20
        print('sampling number in this epoch: %d' % obs.shape[0])

        # update policy-value-network
        n = obs.shape[0]
        advs = returns - values
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
        for _ in range(2):
            ind = np.arange(n)
            np.random.shuffle(ind)
            for batch_id in range(10):
                slices = (tf.constant(arr[ind[batch_id::10]]) for arr in
                          (obs, returns, actions, values, neg_log_p, advs))
                policy_loss, value_loss, entropy, approx_kl, clip_ratio = agent.train(*slices,
                                                                                      lr=lr_now,
                                                                                      clip_range=clip_range_now)

        # logging
        print('epoch-%d, policy loss: %5.3f, value loss: %.5f, entropy: %.5f, approximate kl-divergence: %.5g, '
              'clipped ratio: %.5g' % (update, policy_loss, value_loss, entropy, approx_kl, clip_ratio))
        print('epoch-%d, ave_r: %5.3f, ave_alive: %5.3f, duration: %5.3f' % (
            update, ave_r, np.average(runner.alive_steps_record[-1000:]), time.time() - tick))
        with open(logfile, 'a') as f:
            f.writelines('%d, %.2f, %.2f, %.3f, %.3f, %.3f, %.3f, %.3f, %.2f\n' % (
                update, ave_r, np.average(runner.alive_steps_record[-1000:]), policy_loss, value_loss, entropy,
                approx_kl,
                clip_ratio, time.time() - tick))
        runner.agent.model.model.save(checkpoint_path / ('%d-%.2f' % (update, ave_r)))


if __name__ == '__main__':
    import defopt

    defopt.run(train_senior)
