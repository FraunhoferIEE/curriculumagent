Welcome to CurriculumAgent's documentation!
============================================================

This is the documentation of the CurriculumAgent which is a cleanup and improved version of the
of the `NeurIPS 2020 Competition Agent by binbinchen <https://github
.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution>`_ from the
`L2RPN Competition of 2020 with the Robustness Track <https://competitions.codalab.org/competitions/25426>`_.
The CurriculumAgent has been generalised and should now be able to run on all Grid2Op Environments.

The general idea of the agent is to extract action sets from the Grid2Op environment and then use a rule-based
agent to train
reinforcement learning agent. We explain each step in more detail in our paper
`Managing power grids through topology actions: A comparative study between advanced rule-based and reinforcement learning agents <https://doi.org/10.1016/j.egyai.2023.100276>`_
When using the CurriculumAgent in your work, please cite this paper.

Take a look at the :doc:`overview`, to get a feel of how the agent works and how each part of it can be
used.


.. toctree::
    :maxdepth: 1

    overview
    agents/teacher
    agents/tutor
    agents/junior
    agents/senior
    baseline

    Generated API Documentation<api/curriculumagent>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
