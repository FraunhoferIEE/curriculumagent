
import os
import shutil
from pathlib import Path
import grid2op
import numpy as np
import pytest
from grid2op.Agent import DoNothingAgent

from curriculumagent.tutor.tutor_topology import collect_topology_tutor_one_chronic, generate_topo_tutor_experience, \
    combine_topo_datasets, prepare_topo_dataset
from curriculumagent.tutor.tutors.general_tutor import GeneralTutor


# Idee:
# 1. Wir lauf durch de do nothing durch und nix passiert
# 2. Wenn der Tutor aktiviert wird -> Deep Copy der vect_obs !
# 3. Dann führen wir aus bis
#    a) Tutor nicht mehr gebraucht wird, oder
#    b) wir eine do-nothing action bekommen, weil keine Lösung gefunden wurde -> DN Action
#    c) Actions a1+a2+a3 =A auch abspeichern !
# 4. Zählen wir wie lange der Tutor nicht wieder aktiviert werden muss
# 5. Dann einpflegen der Topology in topo_records. Hier gibt es zwei möglichkeiten:
#    a) Die Topo ist nicht bekannt -> hinzufügen, sowie Anzahl einspeichern
#    b) die topo ist bekannt -> Anzahl hinzuaddieren
# 6. Einpflegen der obs_vect in obs_record + actions
# 7. Final Records zusammenaddieren und rausgeben
# (8.)Aggregation: Die Records müssen nach länge aufgeteilt werden und nach Frequenz analysiert
#    werden. Dann haben wir die obs als Vect, sowie den topo vect



class TestTutorTopology:#
    """
    Testing the Tutor Topology action:
    """

    def test_run_tutor_dn(self,sandbox_env):
        """
        This one is the easiest one, we should get only one topolgy configuration and no obs/act combination:
        """
        obs = sandbox_env.get_obs()
        arr1,arr2 = collect_topology_tutor_one_chronic(action_paths= None,
                                                       chronics_id=0,
                                                       env_name_path="l2rpn_case14_sandbox",
                                                       seed=42,
                                                       enable_logging=True,
                                                       TutorAgent = DoNothingAgent)

        # This should only count the number of steps !
        assert len(arr1) == 1
        assert (arr1[0,1:]==obs.topo_vect.reshape(-1)).all()
        assert arr1[0,0]==1091

        assert arr2.shape == (0,634)


    def test_general_tutor(self,sandbox_env,sandbox_actions):
        """
        Testing with the general Tutor
        """
        obs = sandbox_env.get_obs()
        arr1, arr2 = collect_topology_tutor_one_chronic(action_paths=sandbox_actions,
                                                        chronics_id=0,
                                                        env_name_path="l2rpn_case14_sandbox",
                                                        seed=42,
                                                        enable_logging=True,
                                                        TutorAgent=GeneralTutor)


        assert len(arr1) > 1
        assert arr1.shape[0] < arr2.shape[0]
        assert (arr1[0, 1:] == obs.topo_vect.reshape(-1)).all()
        assert arr1[0, 0] < 1091
        max_id = np.argmax(arr1[:,0])
        # Most prominent is NOT the base topovect:
        assert not (arr1[max_id,1:]==obs.topo_vect.reshape(-1)).all()

    def test_combine_topo_datasets(self,sandbox_env,sandbox_actions):
        """
        Testing, whether the combination worked
        """
        out = []
        for i in [42,100]:
            out.append(collect_topology_tutor_one_chronic(action_paths=sandbox_actions,
                                                            chronics_id=i,
                                                            env_name_path="l2rpn_case14_sandbox",
                                                            seed=42,
                                                            enable_logging=True,
                                                            TutorAgent=GeneralTutor))

        global_topo_id,global_obs_records = combine_topo_datasets(out)


        first = out[0][0]

        # Check if new files were added:
        assert len(first) < len(global_topo_id)

        # The first 45 values should be the same
        assert (first[:, 1:] == global_topo_id[:len(first), 1:]).all()
        # Original Grid should be used more often:
        assert len(np.unique(global_topo_id[:,1:],axis=0)) == len(global_topo_id)

    @pytest.mark.slow
    def test_logic_of_tutor_collect(self,sandbox_env,sandbox_actions,test_temp_save):
        """
        Let's test (relative basic) whether the collect data works
        """
        # Check if dir is empty
        if test_temp_save.is_dir():
            shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)
        # We run with two runs. However, note that the second run only returns
        # the option of do nothing, because it fails quite fast !
        for i in range(2):
            generate_topo_tutor_experience(
                env_name_path="l2rpn_case14_sandbox",
                save_path=test_temp_save,
                action_paths=sandbox_actions,
                num_chronics=i+1,
                num_sample=None,
                jobs=1,
                seed=42,
                TutorAgent=GeneralTutor)

        # Sort Paths by creation date (this should ensure that they are not mixed up):
        paths = sorted(Path(test_temp_save).iterdir(), key=os.path.getmtime)

        dat = [np.load(test_temp_save/a) for a in paths]
        first = dat[0]["topo_id"]
        second = dat[1]["topo_id"]
        # Check if new files were added:
        assert len(first )<len(second)

        # The first 45 values should be the same
        assert (first[:,1:] == second[:len(first),1:]).all()
        # Original Grid should be used more often:
        assert first[0,0] < second[0,0]

        # Now obs/rew:
        assert len(dat[0]["obs_records"])<len(dat[1]["obs_records"])

        del dat

    @pytest.mark.slow
    def test_prepare_topo_dataset(self,test_temp_save,sandbox_env,sandbox_actions):
        """
        This only works if previous test worked.
        """

        obs = sandbox_env.get_obs()

        if test_temp_save.is_dir():
            shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)

        for i in range(2):
            generate_topo_tutor_experience(
                env_name_path="l2rpn_case14_sandbox",
                save_path=test_temp_save,
                action_paths=sandbox_actions,
                num_chronics=i+1,
                num_sample=None,
                jobs=1,
                seed=42,
                TutorAgent=GeneralTutor)

        os.mkdir(test_temp_save/"junior")

        prepare_topo_dataset(traindata_path=test_temp_save,
                             target_path=test_temp_save/"junior",
                             dataset_name="heino",
                             seed=42,
                             top_k_topologies=10,
                             plot_choice=False,
                             env = sandbox_env)

        assert test_temp_save/"junior"/"heino_test.npz"
        assert test_temp_save / "junior" / "heino_train.npz"
        assert test_temp_save / "junior" / "heino_val.npz"
        assert test_temp_save / "junior" / "topologies.npy"

        topo = np.load(test_temp_save / "junior" / "topologies.npy")
        assert topo.shape[0] == 10
        assert topo[0,0] > topo[1,0]
        # Check if base topo is there and only once
        assert (topo[:,1:] == obs.topo_vect).all(axis=1).any()

        with np.load(test_temp_save / "junior" / "heino_train.npz") as train:
            with np.load(test_temp_save / "junior" / "heino_val.npz") as val:
                assert train["s_train"].shape[1] == obs.to_vect().shape[0]
                assert val["s_validate"].shape[1] == obs.to_vect().shape[0]

                topo_v = topo[train["a_train"][0]][:, 1:]
                tr_o = sandbox_env.observation_space.from_vect(train["s_train"][0])
                assert isinstance(tr_o, grid2op.Observation.BaseObservation)

                # Check that the starting topo and the last topo changed
                assert not (topo_v == tr_o.topo_vect).all(axis=0).all()
                assert np.sum((topo_v == tr_o.topo_vect) == False) > 2  # More than two buses were changed

        # Now that the files are closed, you can safely delete the directories
        shutil.rmtree(test_temp_save / "junior")
        shutil.rmtree(test_temp_save)
        os.mkdir(test_temp_save)
