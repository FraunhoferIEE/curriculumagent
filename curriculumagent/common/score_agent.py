"""Script to score and evaluate the final agent and compare it to the original scores.
"""
import json
import logging
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import grid2op
import lightsim2grid
import numpy as np
import pandas as pd
import tensorflow as tf
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from grid2op.Episode import EpisodeData
from grid2op.Reward import L2RPNWCCI2022ScoreFun, L2RPNSandBoxScore
from grid2op.dtypes import dt_int
from grid2op.utils import ScoreL2RPN2020, EpisodeStatistics
from ubelt import Timer

env_name = "l2rpn_neurips_2020_track1"
test_flag = False


@dataclass
class AgentReport:
    """Report that consists of data gathered from evaluating an agent, mainly used for comparing agents and
    tracking performance.
    """

    agent_name: str
    score_data: Dict
    nb_episodes: int
    avg_score: Optional[float] = None
    evaluation_time: Optional[float] = None
    g2op_version: Optional[str] = None

    @staticmethod
    def load(path: Path) -> "AgentReport":
        """Load a report previously saved using :meth:`save`.

        Args:
            path: Path to load the report file from.

        Returns:
            Report.
        """

        with path.open("rb") as f:
            report = pickle.load(f)
        return report

    def save(self, path: Path):
        """Save the report into a machine-readable format for reloading with :meth:`load`.

        Args:
            path: Where to save the report file.

        Returns:
            None.
        """
        with path.open("wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def render_report(report_path: Path, report: AgentReport, comparison_reports: List[AgentReport]):
    """Render the report as a markdown file for easy viewing by humans.

    Args:
        report_path: The path to save the report to.

    Returns:
        None.
    """
    with report_path.open("w") as f:
        f.write(f"# {report.agent_name} Performance Report\n\n")
        f.write(f"**{report.agent_name} Score: {report.avg_score}**\n\n")

        for comparison_report in comparison_reports:
            f.write(f"*{comparison_report.agent_name} Score: {comparison_report.avg_score}*\n\n")

        f.write("\n\n ## Plots\n\n")

        survival_plot = plot_bar_comparison([report] + comparison_reports, "ts_survived")
        score_plot = plot_bar_comparison([report] + comparison_reports, "all_scores")

        survival_plot_name = f"{report.agent_name}_survival.png"
        score_plot_name = f"{report.agent_name}_score.png"
        survival_plot.figure.savefig(report_path.parent / survival_plot_name, bbox_inches="tight")
        score_plot.figure.savefig(report_path.parent / score_plot_name, bbox_inches="tight")

        f.write(f"\n![Score Plot]({score_plot_name})\n")
        f.write(f"\n![Survival Plot]({survival_plot_name})\n")

        f.write("\n\n ## Raw Data/Scores \n\n")
        pd.DataFrame(report.score_data).to_markdown(f)

        f.write(f"\n\n ## Metadata: \n\n Grid2Op Version: {report.g2op_version}")


def plot_bar_comparison(reports: List[AgentReport], metric_key: str) :
    """Create Bar Plot of the performance.

    Args:
        reports: Reports of the agents.
        metric_key: Metric key, where to evaluate.

    Returns:
        Plot.

    """
    assert len(reports) >= 2, "Need at least 2 reports for comparison"

    score_dfs = [(r.agent_name, pd.DataFrame(r.score_data).set_index(("episode_name"))[metric_key]) for r in reports]
    names, comb_series = zip(*score_dfs)
    score_merged = pd.concat(comb_series, axis=1)
    score_merged.columns = names

    axes = (score_merged.plot.bar(title=f"{metric_key} Comparison", ylabel=metric_key, figsize=(10, 6))
            .legend(loc="center left", bbox_to_anchor=(1, 0.5))

            )
    return axes

class ScoreL2RPN2020WithNames(ScoreL2RPN2020):
    """Extension of ScoreL2RPN2020 class to return scenario names as well.

    """
    def get(
            self, agent: BaseAgent, path_save: Optional[str] = None, nb_process: int = 1,
            seed_name:str = None) -> Tuple[List[float], List[int], List[int], List[str]]:
        """Get the score of the agent depending on what has been computed.

        Args:
            agent: The agent you want to score.
            path_save: The path where you want to store the logs of your agent.
            nb_process: Number of process to use for the evaluation.

        Returns:
            all_scores: List of your agent per scenarios score.
            ts_survived: List of the step number your agent successfully managed for each scenario.
            total_ts: Total step number for each scenario.
            scenario_names: Names for each scenario executed.
            seed_name: When running multiple seeds, it is possilbe to specify the name in order to save
            multiple do_nothing reports, depending on the seed of the env.
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

        es_instance = EpisodeStatistics(self.env, name_stats=seed_name)

        es_instance.run_env(
            self.env,
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
            score_this_ep, nb_ts_survived, total_ts_tmp = self._compute_episode_score(
                ep_id,
                meta=this_epi_meta,
                other_rewards=this_epi_scores,
                dn_metadata=meta_data_dn,
                no_ov_metadata=no_ov_metadata,
            )
            all_scores.append(score_this_ep)
            ts_survived.append(nb_ts_survived)
            total_ts.append(total_ts_tmp)
            scenario_names.append(this_ep_nm)

        if need_delete:
            dir_tmp.cleanup()
        return all_scores, ts_survived, total_ts, scenario_names


class ScoreL2RPN2022WithNames(ScoreL2RPN2020WithNames):
    """Extension of ScoreL2RPN2020WithNames class with the initialisation of the ScoreL2RPN2022 values.

    The only difference is that it takes a different score function. See `ScoreL2RPN2022
    <https://grid2op.readthedocs.io/en/latest/_modules/grid2op/utils/l2rpn_wcci_2022_scores.html#ScoreL2RPN2022>`_
    """

    def __init__(self,
                 env,
                 env_seeds=None,
                 agent_seeds=None,
                 nb_scenario=16,
                 min_losses_ratio=0.8,
                 verbose=0, max_step=-1,
                 nb_process_stats=1,
                 scores_func=L2RPNWCCI2022ScoreFun,
                 score_names=None):
        super().__init__(env, env_seeds, agent_seeds, nb_scenario, min_losses_ratio, verbose, max_step,
                         nb_process_stats, scores_func, score_names)


def score_agent(
        agent: BaseAgent,
        env: BaseEnv,
        log_path: Path,
        seed: int = 42,
        name: Optional[str] = None,
        nb_episodes: Optional[int] = None,
        nb_process: int = os.cpu_count(),
        reinit: Optional[bool] = False,
        score_l2rpn2020: Optional[bool] = True
) -> AgentReport:
    """Score the given agent in the given environment, saving logs in log_dir.

    Args:
        agent: The agent to score.
        env: The environment to score the agent on.
        log_path: The directory to save agent logs in.
        seed: Seed used to score the agent.
        name: The name of the agent.
        nb_episodes: How many episodes/chronics the agent should be scored. Equals 2 by default.
        nb_process: How many processes should be used to perform the scoring.
                    Some agents may support one process only due to unsupported serialization/pickling of state.
        reinit: Optional Boolean, whether the underlying environment statistics should be deleted and
                recalculated.
        score_l2rpn2020: Optional Boolean Indicating, which score should be used for the evaluation. If True the
                score_agent method will use the ScoreL2RPN2020 metric, else ScoreL2RPN2022.

    Returns:
        The :class:`AgentReport` describing the agent performance.

    """
    g2op_version = str(grid2op.__version__)
    ls2g_version = str(lightsim2grid.__version__)
    print(f"Grid2Op version: {g2op_version}")
    print(f"lightsim2grid version: {ls2g_version}")
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != "GPU"
    except Exception as e:
        # Invalid device or cannot modify virtual devices once initialized.
        logging.warning(f"We caught the exception {e}")
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
        agent_name = f"{name}_{agent.__class__.__name__}"
    else:
        agent_name = f"{agent.__class__.__name__}"
    print(f"Evaluating agent with name {agent_name}")

    # Start scoring the agent and time the execution
    timer = Timer()
    with timer:
        if reinit:
            li_stats = EpisodeStatistics.list_stats(env)
            print(f"Deleting the following environment statistics for seed {seed}:")
            print(li_stats)

            for path, el in li_stats:
                shutil.rmtree(os.path.join(path, el), ignore_errors=True)
            print("Delition done! Start re-running agent")

        if score_l2rpn2020:
            my_score = ScoreL2RPN2020WithNames(
                env, nb_scenario=nb_episodes, env_seeds=env_seeds, agent_seeds=agent_seeds, verbose=3
            )
        else:
            my_score = ScoreL2RPN2022WithNames(
                env, nb_scenario=nb_episodes, env_seeds=env_seeds, agent_seeds=agent_seeds, verbose=3
            )

        log_path.mkdir(exist_ok=True, parents=True)
        all_scores, ts_survived, total_ts, episode_names = my_score.get(
            agent, nb_process=nb_process, path_save=str(log_path),seed_name=str(seed)
        )

    score_data = {
        "all_scores": all_scores,
        "ts_survived": ts_survived,
        "total_ts": total_ts,
        "episode_name": episode_names,
    }
    avg_score = np.array(all_scores).mean()
    return AgentReport(
        agent_name=agent_name,
        score_data=score_data,
        nb_episodes=nb_episodes,
        evaluation_time=timer.elapsed,
        g2op_version=str(grid2op.__version__),
        avg_score=avg_score,
    )


def load_or_run(
        agent: BaseAgent,
        env: BaseEnv,
        output_path: Path,
        name: Optional[str] = None,
        nb_processes: int = os.cpu_count(),
        overwrite: bool = False,
        number_episodes: int = 2,
        seed: int = 42,
        reinit: bool = False,
        score_l2rpn2020: Optional[bool] = True
) -> AgentReport:
    """Load the given report at cache_path or score the agent if it doesn't exist.

    For easier comparison, this function tries to load cached results of the :func:`score_agent` function.
    However, it does not check whether the result is out of date. So if you updated the agent to score, you either have
    to delete the corresponding cache file or set the overwrite parameter to True.

    Args:
        agent: The agent to score.
        env: The environment to score the agent in.
        output_path: The path to a directory where cached results as well as agent logs of the scoring will be stored.
        name: The name of the report. The agent class name will be used if this is not set.
        nb_processes: How many processes should be used to perform the scoring.
            Some agents may only support one process due to unsupported serialization/pickling of state.
        overwrite: If set to true, ignore and overwrite the cached result.
        number_episodes: How many episodes to run for comparison.
        seed: Optional seed.
        reinit: Optional Boolean, whether the underlying environment statistics should be deleted and
                recalculated.
        score_l2rpn2020: Optional Boolean, indicating which score should be used for the evaluation. If True the
                score_agent() method will use the ScoreL2RPN2020 metric, else the ScoreL2RPN2022.

    Returns:
        The :class:`AgentReport` describing the agent performance.

    """
    if not name:
        name = agent.__class__.__name__
    output_path.mkdir(exist_ok=True, parents=True)
    report_path = output_path / Path(f"{name}_report_data.pkl")
    if report_path.exists() and not overwrite and not reinit:
        logging.info(f"Using cached results from {report_path}")
        report = AgentReport.load(report_path)
        return report

    logs_path = output_path / "agent_logs" / f"{name}"

    report = score_agent(
        agent, env, log_path=logs_path, name=name, nb_process=nb_processes, nb_episodes=number_episodes,
        seed=seed, reinit=reinit, score_l2rpn2020=score_l2rpn2020
    )
    report.save(report_path)
    return report
