HUGO -- Highlighting Unseen Grid Options: Combining Deep Reinforcement Learning with a Heuristic Target Topology Approach
===============

This is a very short ReadMe regarding  our HUGO
[Paper](https://arxiv.org/abs/2405.00629#). As the Curriculum Agent is always a continous process, including 
updates and changes in the code base. Therefore, we would propose to recreate the code with the following version: 

```
pip install curriculumagent==1.1.0
```
When using the content of this paper, please cite: 
```
@article{lehna2024hugo,
  title={HUGO--Highlighting Unseen Grid Options: Combining Deep Reinforcement Learning with a Heuristic Target Topology Approach},
  author={Lehna, Malte and Holzh{\"u}ter, Clara and Tomforde, Sven and Scholz, Christoph},
  journal={arXiv preprint arXiv:2405.00629},
  year={2024}
}
```

Structure
-------
The structure of the paper data is as follows: 
- ```evaluation```: Here is the notebook and the files (as pickle) to run the evaluation of the paper. As the 
  overall data was too large, we could only provide the pickled results. 
- ```files```: We provide the newly trained RL Agent on the WCCI 2022 grid with the 2030 actions, aswell as the 500 
  target topologies. 
- ```scripts```: In this directory, you have the scripts to execute the evaluation 
- ```paper_agents.py```: In this file, you find the Topology agent 

License
-------

```
Copyright (c) Fraunhofer IEE
The code is subject to the terms of Mozilla Public License (MPL) v2.0.
Commercial use is NOT allowed.
```

Please take a look at the LICENSE file for a full copy of the MPL license.
