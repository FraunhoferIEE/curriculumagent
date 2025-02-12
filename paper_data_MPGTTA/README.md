Managing power grids through topology actions: A comparative study between advanced rule-based and reinforcement learning agents
===============

This is a very short ReadMe regarding  our 
[Paper](https://www.sciencedirect.com/science/article/pii/S2666546823000484). As the Curriculum Agent is always a continous process, including 
updates and changes in the code base. Therefore, we would propose to recreate the code with the following version: 

```
pip install curriculumagent==1.0.2
```
When using the CurriculumAgent, please cite our paper with.
```
@article{lehna_managing_2023,
	title = {Managing power grids through topology actions: A comparative study between advanced rule-based and reinforcement learning agents},
	issn = {2666-5468},
	url = {https://www.sciencedirect.com/science/article/pii/S2666546823000484},
	doi = {https://doi.org/10.1016/j.egyai.2023.100276},
	pages = {100276},
	journaltitle = {Energy and {AI}},
	author = {Lehna, Malte and Viebahn, Jan and Marot, Antoine and Tomforde, Sven and Scholz, Christoph},
	date = {2023},
}
```

Structure
-------
The structure of the paper data is as follows: 
- ```action_spaces_paper```: Here, we have the two action spaces from Binbinchen and our new N-1 Actions
- ```checkpoint```: In this directory, you can find the model of the Agent that we used for later evaluation
- ```scripts_paper```: Here, there are all scripts of the paper to run all the different agents. 
- In the remaing directory, we have the evaluation code in ```Code for paper.ipynb``` and all the variables 
  in pickle format or as json 

License
-------

```
Copyright (c) Fraunhofer IEE
The code is subject to the terms of Mozilla Public License (MPL) v2.0.
Commercial use is NOT allowed.
```

Please take a look at the LICENSE file for a full copy of the MPL license.
