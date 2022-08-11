"""
Script to score and evaluate the final agent and compare it to the original scores.
"""
import json
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import grid2op
import lightsim2grid
import numpy as np
import pandas as pd
import tensorflow as tf
from grid2op.Agent import BaseAgent, DoNothingAgent
from grid2op.Environment import BaseEnv
from grid2op.Episode import EpisodeData
from grid2op.Reward import RedispReward, L2RPNSandBoxScore
from grid2op.dtypes import dt_int
from grid2op.utils import ScoreL2RPN2020, EpisodeStatistics
from lightsim2grid import LightSimBackend
from matplotlib.axes._base import _AxesBase
from ubelt import Timer

from curriculumagent.tutor.tutors import original_tutor

env_name = "l2rpn_neurips_2020_track1"
test_flag = False


@dataclass
class AgentReport:
    """
    Report that consists of data gathered from evaluating an agent, mainly used for comparing agents and
    tracking performance.
    """

    agent_name: str
    score_data: Dict
    nb_episodes: int
    avg_score: Optional[float] = None
    evaluation_time: Optional[float] = None
    g2op_version: Optional[str] = None

    @staticmethod
    def load(path: Path) -> 'AgentReport':
        """
        Load a report previously saved using :meth:`save`.

        Args:
            path: Path to load the report file from.
        """

        with path.open('rb') as f:
            report = pickle.load(f)
        return report

    def save(self, path: Path):
        """
        Save the report into a machine readable format for reloading with :meth:`load`.

        Args:
            path: Where to save the report file.
        """
        with path.open('wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def render_report(report_path: Path, report: AgentReport, comparison_reports: List[AgentReport]):
    """
    Render the report as a markdown file for easy viewing by humans.

    Args:
        report_path: The path to save the report to.
    """
    with report_path.open('w') as f:
        f.write(f"# {report.agent_name} Performance Report\n\n")
        f.write(f"**{report.agent_name} Score: {report.avg_score}**\n\n")

        for comparison_report in comparison_reports:
            f.write(f"*{comparison_report.agent_name} Score: {comparison_report.avg_score}*\n\n")

        f.write("\n\n ## Plots\n\n")

        survival_plot = plot_bar_comparison([report] + comparison_reports, 'ts_survived')
        score_plot = plot_bar_comparison([report] + comparison_reports, 'all_scores')

        survival_plot_name = f'{report.agent_name}_survival.png'
        score_plot_name = f'{report.agent_name}_score.png'
        survival_plot.figure.savefig(survival_plot_name)
        score_plot.figure.savefig(score_plot_name)

        f.write(f"\n![Score Plot]({score_plot_name})\n")
        f.write(f"\n![Survival Plot]({survival_plot_name})\n")

        f.write("\n\n ## Raw Data/Scores \n\n")
        pd.DataFrame(report.score_data).to_markdown(f)

        f.write(f"\n\n ## Metadata: \n\n Grid2Op Version: {report.g2op_version}")


def plot_bar_comparison(reports: List[AgentReport], metric_key: str) -> _AxesBase:
    assert len(reports) >= 2, 'Need at least 2 reports for comparison'

    score_dfs = [(r.agent_name, pd.DataFrame(r.score_data).set_index(('episode_name'))) for r in reports]

    metric_keys = [f'{score_dfs[0][0]}']
    score_merged = score_dfs[0][1].rename(columns={metric_key: metric_keys[0]})
    for agent_name, score_df in score_dfs[1:]:
        new_key = f'{agent_name}'
        score_merged = pd.merge(score_merged, score_df.rename(columns={metric_key: new_key}), left_index=True,
                                right_index=True)
        metric_keys.append(new_key)

    axes = score_merged[metric_keys].plot.bar(title=f'{metric_key} Comparison', ylabel=metric_key, figsize=(10, 6))
    return axes


class ScoreL2RPN2020WithNames(ScoreL2RPN2020):
    """Extension of ScoreL2RPN2020 class to also return scenario names."""

    def get(self, agent: BaseAgent, path_save: Optional[str] = None, nb_process: int = 1) \
            -> Tuple[List[float], List[int], List[int], List[str]]:
        """
        Get the score of the agent depending on what has been computed.

        Parameters
        ----------
        agent: :class:`grid2op.Agent.BaseAgent`
            The agent you want to score

        path_save: ``str``
            the path were you want to store the logs of your agent.

        nb_process: ``int``
            Number of process to use for the evaluation

        Returns
        -------
        all_scores: ``list``
            List of the score of your agent per scenarios

        ts_survived: ``list``
            List of the number of step your agent successfully managed for each scenario

        total_ts: ``list``
            Total number of step for each scenario

        scenario_names: ``list``
            Names for each scenario executed
        """
        if path_save is not None:
            need_delete = False
            path_save = os.path.abspath(path_save)
        else:
            need_delete = True
            dir_tmp = tempfile.TemporaryDirectory()
            path_save = dir_tmp.name

        if self.verbose >= 1:
            print("Starts the evaluation of the agent")
        EpisodeStatistics.run_env(self.env,
                                  env_seeds=self.env_seeds,
                                  agent_seeds=self.agent_seeds,
                                  path_save=path_save,
                                  parameters=self.env.parameters,
                                  scores_func=self.scores_func,
                                  agent=agent,
                                  max_step=self.max_step,
                                  nb_scenario=self.nb_scenario,
                                  pbar=self.verbose >= 2,
                                  nb_process=nb_process,
                                  )
        if self.verbose >= 1:
            print("Start the evaluation of the scores")

        meta_data_dn = self.stat_dn.get_metadata()
        no_ov_metadata = self.stat_no_overflow_rp.get_metadata()

        all_scores = []
        ts_survived = []
        total_ts = []
        scenario_names = []
        for ep_id in range(self.nb_scenario):
            this_ep_nm = meta_data_dn[f"{ep_id}"]["scenario_name"]
            with open(os.path.join(path_save, this_ep_nm, EpisodeData.META), "r", encoding="utf-8") as f:
                this_epi_meta = json.load(f)
            with open(os.path.join(path_save, this_ep_nm, EpisodeData.OTHER_REWARDS), "r", encoding="utf-8") as f:
                this_epi_scores = json.load(f)
            score_this_ep, nb_ts_survived, total_ts_tmp = \
                self._compute_episode_score(ep_id,
                                            meta=this_epi_meta,
                                            other_rewards=this_epi_scores,
                                            dn_metadata=meta_data_dn,
                                            no_ov_metadata=no_ov_metadata)
            all_scores.append(score_this_ep)
            ts_survived.append(nb_ts_survived)
            total_ts.append(total_ts_tmp)
            scenario_names.append(this_ep_nm)

        if need_delete:
            dir_tmp.cleanup()
        return all_scores, ts_survived, total_ts, scenario_names


def score_agent(agent: BaseAgent, env: BaseEnv, log_path: Path,
                seed: int = 42,
                name: Optional[str] = None, nb_episodes: Optional[int] = None,
                nb_process: int = os.cpu_count()) -> AgentReport:
    """
    Score the given agent in the given environment, saving logs in log_dir.

    Args:
        agent: The agent to score.
        env: The environment to score the agent in.
        log_path: The directory to save agent logs in.
        seed: Seed used to score the agent.
        name: The name of the agent.
        nb_episodes: How many episodes/chronics the agent should be scored. 2 by default.
        nb_process: How many processes should be used to perform the scoring.
                    Some agents may only support 1 process due to unsupported serialization/pickling of state.

    Returns: The :class:`AgentReport` describing the agents performance.

    """
    g2op_version = str(grid2op.__version__)
    ls2g_version = str(lightsim2grid.__version__)
    print(f"Grid2Op version: {g2op_version}")
    print(f"lightsim2grid version: {ls2g_version}")
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # Set random Seed
    if not nb_episodes:
        nb_episodes = 2
    np.random.seed(seed)
    max_int = np.iinfo(dt_int).max
    env_seeds = list(np.random.randint(max_int, size=nb_episodes))
    agent_seeds = list(np.random.randint(max_int, size=nb_episodes))

    # Evaluate trained agent
    if name:
        agent_name = f'{name}_{agent.__class__.__name__}'
    else:
        agent_name = f'{agent.__class__.__name__}'
    print(f"Evaluating agent with name {agent_name}")

    # Start scoring the agent and time the execution
    timer = Timer()
    with timer:
        my_score = ScoreL2RPN2020WithNames(env,
                                           nb_scenario=nb_episodes,
                                           env_seeds=env_seeds,
                                           agent_seeds=agent_seeds,
                                           verbose=3)

        log_path.mkdir(exist_ok=True, parents=True)
        all_scores, ts_survived, total_ts, episode_names = my_score.get(agent, nb_process=nb_process,
                                                                        path_save=str(log_path))

    score_data = {
        'all_scores': all_scores,
        'ts_survived': ts_survived,
        'total_ts': total_ts,
        'episode_name': episode_names,
    }
    avg_score = np.array(all_scores).mean()
    return AgentReport(agent_name=agent_name, score_data=score_data, nb_episodes=nb_episodes,
                       evaluation_time=timer.elapsed, g2op_version=str(grid2op.__version__), avg_score=avg_score)


def load_or_run(agent: BaseAgent, env: BaseEnv, output_path: Path,
                name: Optional[str] = None, nb_processes: int =os.cpu_count(), overwrite: bool=False) -> AgentReport:
    """
    Load the given report at cache_path or score the agent if it doesn't exist.

    For easier comparison this function tries to load cached results of the :func:`score_agent` function.
    However it does not check if the result is out of date. So if you updated the agent to score you either have
    to delete the corresponding cache file or set the overwrite parameter to true.

    Args:
        agent: The agent to score.
        env: The environment to score the agent in.
        output_path: The path to a directory where cached results and agent logs of the scoring will be stored.
        name: The name of the report. The agents class name will be used if this is not set.
        nb_process: How many processes should be used to perform the scoring.
            Some agents may only support 1 process due to unsupported serialization/pickling of state.
        overwrite: If set to true, ignore and overwrite the cached result.

    Returns: The :class:`AgentReport` describing the agents performance.

    """
    if not name:
        name = agent.__class__.__name__
    output_path.mkdir(exist_ok=True, parents=True)
    report_path = output_path / Path(f'{name}_report_data.pkl')
    if report_path.exists() and not overwrite:
        logging.info(f'Using cached results from {report_path}')
        report = AgentReport.load(report_path)
        return report

    logs_path = output_path / 'agent_logs' / f'{name}'

    report = score_agent(agent, env, log_path=logs_path, name=name, nb_process=nb_processes)
    report.save(report_path)
    return report


if __name__ == '__main__':
    log_format = '(%(asctime)s) [%(name)-10s] %(levelname)8s: %(message)s [%(filename)s:%(lineno)s]'
    logging.basicConfig(level=logging.WARN, format=log_format)

    reward = RedispReward
    other_rewards = {'grid_operation_cost': L2RPNSandBoxScore}

    env_path = str(Path(__file__).parent.parent.parent / "training_data_track1")
    env: BaseEnv = grid2op.make(env_path, test=test_flag,
                                backend=LightSimBackend(), reward_class=reward, other_rewards=other_rewards)

    output_path = Path('score_output')
    do_nothing_agent = DoNothingAgent(env.action_space)
    dn_report = load_or_run(do_nothing_agent, env, output_path, name="DoNothing")

    old_action_space_path = Path(__file__).parent.parent / 'action_space'
    tutor_original = original_tutor.Tutor(env.action_space, old_actionspace_path=old_action_space_path)
    tutor_original_report = load_or_run(tutor_original, env, output_path, name="Tutor Original")

    render_report(Path('report.md'), tutor_original_report, [ dn_report])