CurriculumAgent
===============

CurriculumAgent is a Reinforcement Learning Agent designed to learn from and act within the  
[Grid2Op Environments](https://grid2op.readthedocs.io/en/latest/). Originated from the [NeurIPS 2020 Competition 
Agent by binbinchen](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution), this package contains a cleanup 
and improved version of the agent. Next to the general code cleanup, the package is updated for the newer Grid2Op 
version and future research results are planed to be published for this package. 

[![CI](https://github.com/FraunhoferIEE/CurriculumAgent/actions/workflows/main.yml/badge.svg)](https://github.com/FraunhoferIEE/CurriculumAgent/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/curriculumagent/badge/?version=latest)](https://curriculumagent.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/CurriculumAgent.svg)](https://badge.fury.io/py/CurriculumAgent)


Setup
-----

All requirements are listed in `requirements.txt`. Installing the package should already give you all needed 
requirements. Furthermore, the package can also be installed via
```{python}
pip install CurriculumAgent
```


Usage/Documentation
-------------------

Please take a look at our [sphinx documentation](https://curriculumagent.readthedocs.io/en/latest/) on how to use the package.

We also provide several jupyter notebooks in `./jupyter_notebooks` to get you started quickly.

In addition, a the Agent will be added to the [L2RPN Baselines](https://github.com/rte-france/l2rpn-baselines/tree/master/l2rpn_baselines) in the next few weeks. 

License
-------

```
Copyright (c) 2022 EI Innovation Lab, Huawei Cloud, Huawei Technologies and Fraunhofer IEE
The code is subject to the terms of Mozilla Public License (MPL) v2.0.
Commercial use is NOT allowed.
```

Please take a look at the LICENSE file for a full copy of the MPL license.
