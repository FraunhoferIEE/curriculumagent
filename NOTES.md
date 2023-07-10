Notes about running binbinchen
==============================

- All steps could be run in isolation with the provided intermediate results
- They provide a demo dataset/setup and mention that we should use our own data

Fix for newer Grid2OP version(zerofix)
--------------------------------------

To make the agent work on newer version of Grid2Op every line
that says

    np.zeros_like(obs.rho)

has to be replaced by

    np.zeros_like(obs.rho, dtype=int)

This was tested after the fix with `Grid2Op==1.9.0`.

Further, this agent was original build for the 2020 challenge, however, should now work for all 
Gri2Op environments. 

Teacher
-------

- Teacher1.py: runs fine with reduced NUM_EPISODES = 1(line 78) and zerofix
  - Generates `Teacher/Experiences1.csv` file
    
- Teacher2.py: runs fine with reduced NUM_EPISODES = 1(line 114) and zerofix
  - Generates `Teacher/Experiences2.csv` file
    
- Generate_action_space.py: runs fine without modification
  - Takes the two previously generated csv files
  - Generates `ActionSpace/actions62.npy` and `ActionSpace/actions146.npy` 
    **probably only when run with the right paramaters** depending on how the input files look
    - When running with the reduced episodes it produced an `actions4.npy`
  
- Each iteration takes rather long because of the topological search of all actions
    
Tutor
-----

- Generate_teaching_dataset.py
  - Takes `ActionSpace/actions62.npy` and `ActionSpace/actions146.npy`(**doesn't support actions4.npy out of the box**)
  - Generates `JuniorStudent/TrainingData/records*.npy` file for supervised training

- Tutor.py
  - Merely contains the Teacher implementation which gets imported by Generate_teaching_dataset.py

JuniorStudent
-------------

- JuniorStudent.py
  - The task at `JuniorStudent.py:107` has to be changed from
    `task='Convert'` to `task='Train'`, then Test and then Convert.
  - Takes the `'JuniorStudent/TrainingData/records_11-13-11-18.npy'` file to pretrain the provided network.
    - This has to be changed for self generated records.
  - Generates `SeniorModel/JuniorModel/` with weights from the neural pretrained neural network

SeniorStudent
-------------

- SeniorStudent.py: runs fine with reduced EPOCHS=1(line 175) and
  created `SeniorStudent/log` directory which was not present and caused an error.
  - Takes the `SeniorModel/JunioModel/*` files for further training
- PPO.py: Definition of the network matching the pretrained one
- PPO_Reward.py: Definition of the reward function

