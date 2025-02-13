import os
import logging
from pathlib import Path
from curriculumagent.tutor.tutor import general_tutor, n_minus_1_tutor


def main():
    DATA_PATH = Path("l2rpn_neurips_2020_track1_small")

    SAVE_PATH = Path('n1_topo')
    ACTION_SPACE_DIRECTORY = Path('action_spaces_paper')
    NUM_CHRONICS = 5000
    TOTAL_CHRONICS = len(os.listdir(DATA_PATH / 'chronics'))
    # Path(ACTION_SPACE_DIRECTORY) / "new" / "actionspace_tuples.npy",
    actions_list = [ACTION_SPACE_DIRECTORY / "n-1_actions.npy",
                    ACTION_SPACE_DIRECTORY / "actions62.npy",
                    ACTION_SPACE_DIRECTORY / "actions146.npy"]

    print(TOTAL_CHRONICS)
    for seed in range(300):
        print(f"Run new iteration with seed: {seed}")

        n_minus_1_tutor(env_name_path=DATA_PATH,
                        action_paths=actions_list,
                        save_path=SAVE_PATH,
                        num_chronics=TOTAL_CHRONICS,
                        jobs=-1,
                        num_sample=NUM_CHRONICS, seed=seed,
                        revert_to_original_topo=True)


if __name__ == "__main__":
    main()
