CurriculumAgent
===============
[![CI](https://github.com/FraunhoferIEE/CurriculumAgent/actions/workflows/main.yml/badge.svg)](https://github.com/FraunhoferIEE/CurriculumAgent/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/curriculumagent/badge/?version=latest)](https://curriculumagent.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/curriculumagent.svg)](https://badge.fury.io/py/curriculumagent)

CurriculumAgent is a cleanup and improved version of the
[NeurIPS 2020 Competition Agent by binbinchen](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution).
The agent is build to extract action sets of the Grid2Op Environment and then use rule-based agent to train
a Reinforcement Learning agent. We explain each step in more detail in our paper. 

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
The code of the paper can be found under `/paper_data_MPGTTA`.

Setup
-----

All requirements are listed in `requirements.txt`.

Installing the package should already give you all needed requirements.

Usage/Documentation
-------------------

Please take a look at our [sphinx documentation](https://curriculumagent.readthedocs.io/en/latest/) on how to use the package.

We also provide several jupyter notebooks in `./jupyter_notebooks` to get you started quickly.



License
-------

```
Copyright (c) 2022 EI Innovation Lab, Huawei Cloud, Huawei Technologies and Fraunhofer IEE
The code is subject to the terms of Mozilla Public License (MPL) v2.0.
Commercial use is NOT allowed.
```

Please take a look at the LICENSE file for a full copy of the MPL license.
