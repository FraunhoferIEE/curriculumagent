Fault Detection for agents on power grid topology optimization: A Comprehensive analysis
===============

This is a very short ReadMe regarding  our Fault Detection
[Paper](https://arxiv.org/abs/2406.16426). As the Curriculum Agent is always a continous process, including 
updates and changes in the code base. Therefore, we would propose to recreate the code with the following version: 

```
pip install curriculumagent==1.1.0
```
When using the content of this paper, please cite: 
```
@article{lehna2024fault,
  title={Fault Detection for agents on power grid topology optimization: A Comprehensive analysis},
  author={Lehna, Malte and Hassouna, Mohamed and Degtyar, Dmitry and Tomforde, Sven and Scholz, Christoph},
  journal={arXiv preprint arXiv:2406.16426},
  year={2024}
}
```

Structure
-------
The structure of the paper data is as follows: 
- ```data_gen```: This is the script to get the evaluation of the agents on the WCCI 2022 grid 
- ```models```: Here are all remaining data results and models
- ```notebooks_&_scripts```: Here you find all Notebooks and training scripts to run the clustering and forecasting

License
-------

```
Copyright (c) Fraunhofer IEE
The code is subject to the terms of Mozilla Public License (MPL) v2.0.
Commercial use is NOT allowed.
```

Please take a look at the LICENSE file for a full copy of the MPL license.
