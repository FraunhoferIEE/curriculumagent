import pytest
import os
import matplotlib
import shutil
from pathlib import Path
from curriculumagent.common.score_agent import AgentReport, render_report, plot_bar_comparison, load_or_run
from grid2op.Agent import DoNothingAgent
import pytest
import pickle

test_flag = False


class TestLoadOrRun:
    """
    Testing the load or run
    """

    @pytest.mark.slow
    def test_load_or_run(self, test_env, test_temp_save):
        """
        Check if temporary save is empty, if not delete old files
        Run the report and check if output correct
        Check if directory agent_logs and agent_logs/DoNothing exist
        Check if the file DoNothing_report_data.pkl was created
        assert score of the DoNothing agent with dn_report.avg_score
        assert name of the agent
        DO NOT Delete the files
        """

        do_nothing_agent = DoNothingAgent(test_env.action_space)
        data_path = test_temp_save
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if len(os.listdir(data_path)):
            for root, dirs, files in os.walk(data_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)

        dn_report = load_or_run(do_nothing_agent, test_env, data_path, name="DoNothing", number_episodes=1,
                                nb_processes=1)
        assert isinstance(dn_report, AgentReport)

        assert (data_path.joinpath('agent_logs')).is_dir()
        assert (data_path.joinpath('agent_logs').joinpath('DoNothing')).is_dir()
        assert os.path.exists(data_path.joinpath('DoNothing_report_data.pkl'))
        assert dn_report.avg_score == 0.0
        assert dn_report.agent_name == 'DoNothing_DoNothingAgent'

    def test_load(self, test_temp_save):
        '''
        Copy the pickled file to a new directory
        Check if the file can be loaded with load_or_run and the report
        Check how the report looks like.
        '''
        data_path = test_temp_save
        source_path = data_path.joinpath('DoNothing_report_data.pkl')
        target_path_with_file = Path(__file__).parent / "data" / "report" / "DoNothing_report_data.pkl"
        target_path = Path(__file__).parent / "data" / "report"
        os.makedirs(target_path, exist_ok=True)
        shutil.copy2(source_path, target_path_with_file)
        try:
            report = AgentReport.load(Path(__file__).parent / "data" / "report" / "DoNothing_report_data.pkl")
            assert report.avg_score == 0.0
            assert report.agent_name == 'DoNothing_DoNothingAgent'
        except pickle.UnpicklingError:
            pytest.raises(pickle.UnpicklingError)

    def test_plot_bar_comparison(self,test_temp_save):
        '''
        Plot function: Load the Report (similar to second test) and then execute the plot method
        Check if it is a plot.
        Check the Legend, axis
        '''
        report = AgentReport.load(Path(__file__).parent / "data" / "report" / "DoNothing_report_data.pkl")
        axis = plot_bar_comparison([report, report], 'all_scores')
        assert isinstance(axis.legendHandles[0], matplotlib.patches.Rectangle)
        assert isinstance(axis.legendHandles[1], matplotlib.patches.Rectangle)
        assert axis.axes.name == 'rectilinear'


    def test_render_report(self,test_temp_save):
        """
        Testing, whether the rendering of the report works
        """
        report_path = test_temp_save.parent / "report_data" / "DoNothing_report_data.pkl"
        report = AgentReport.load(report_path)
        render_report(test_temp_save / 'report.md', report, [report])
        assert (test_temp_save / "DoNothing_DoNothingAgent_score.png").is_file()
        assert (test_temp_save / "DoNothing_DoNothingAgent_survival.png").is_file()

        assert (test_temp_save / 'report.md').is_file()

        # Delete and create
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)