import os
import shutil

import pytest
from nni.experiment import ExperimentConfig

from curriculumagent.junior.hyper_parameter_search.hyper_parameter_search_junior import get_default, \
    run_one_experiment, run_nni_experiment


class TestExecution:
    """
    Testing, whether the overall execution works
    """

    def test_get_default(self):
        """
        Testing the default values
        """
        expected = {
            "epochs": 10,  # set to 1000, should be adjusted by the A. hyperband
            "learning_rate": 1e-5,
            "activation": "relu",
            "layer1": 1000,
            "layer2": 1000,
            "layer3": 1000,
            "layer4": 1000,
            "dropout1": 0.25,
            "dropout2": 0.25,
            "patience": 0,
            "initializer": "O",
            'batchsize': 256
        }
        default = get_default()
        for k, v in expected.items():
            assert default[k] == v

    def test_run_one_experiment(self,test_submission_action_space, test_path_data):
        """
        Testing, if the model run works
        """
        test_data = test_path_data / "junior_experience"
        config = get_default()
        acc = run_one_experiment(config=config, action_space_file=test_submission_action_space, path_to_files=test_data, run_nni=False)

        assert isinstance(acc,float)
        assert 0.0 <= acc <= 1.0


class TestNNIHyperOpt:
    """
    A small test that checks, whether the hyper-optimization method works. Note that this only checks
    for the default values and only runs on experiment.
    """

    def test_config(self, test_path_data):
        """
        Testing, whether the creation of the config worked.
        """
        dir_of_j = test_path_data.parent.parent / "curriculumagent" / "junior" / "hyper_parameter_search"
        exp_config = ExperimentConfig(experiment_name='Junior',
                                      experiment_type="hpo",
                                      trial_concurrency=4,
                                      max_experiment_duration='48h',
                                      max_trial_number=1,
                                      training_service_platform='local',
                                      search_space_file=str(dir_of_j / 'search_space.json'),
                                      use_annotation=False,
                                      advisor={'name': 'BOHB',
                                               'class_args': {'optimize_mode': 'maximize',
                                                              'min_budget': 1,
                                                              'max_budget': 100,
                                                              'eta': 3,
                                                              'min_points_in_model': 17,
                                                              'top_n_percent': 20,
                                                              'num_samples': 128,
                                                              'random_fraction': 0.33,
                                                              'bandwidth_factor': 3.0,
                                                              'min_bandwidth': 0.001}},
                                      trial_command= 'hyper_parameter_search_junior.py',
                                      trial_code_directory=dir_of_j ,
                                      trial_gpu_number=0)
        assert isinstance(exp_config, ExperimentConfig)
        exp_config.validate()

    @pytest.mark.ultra_slow
    @pytest.mark.slow
    def test_run_nni_experiment(self, test_temp_save, test_path_data):
        """
        Testing if the hyper-optimization works with NNI. This is only the check, whether the initialization of the
        experiment works. This might still not mean that the hyper-paramter search worked. More test come with
        NNI=3.0

        This test is still experimental
        """

        # Check if dir is empty
        if test_temp_save.is_dir():
            shutil.rmtree(test_temp_save, ignore_errors=True)

        os.mkdir(test_temp_save)

        assert len(os.listdir(test_temp_save)) == 0
        old_dir = os.getcwd()

        test_data = test_path_data / "junior_experience"
        assert os.getcwd()
        run_nni_experiment(test_data,
                           max_trial_number=1,
                           experiment_working_directory=test_temp_save)
        dir_content = os.listdir(test_temp_save)

        # the run_nni_experiment changes the work directory.
        assert os.getcwd() == old_dir
        assert len(dir_content) > 0
        # Technically latest could also work
        assert (test_temp_save / dir_content[0]/ "checkpoint").is_dir()
        # Delete after working:
        shutil.rmtree(test_temp_save, ignore_errors=True)
