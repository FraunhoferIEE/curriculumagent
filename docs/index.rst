Welcome to CurriculumAgent's documentation!
============================================================

This is the documentation of the CurriculumAgent which is a fork and improvement
of the `binbinchen agent <https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution>`_ of the
`L2RPN Competition of 2020 with the Robustness Track <https://competitions.codalab.org/competitions/25426>`_.

The main improvements over the original agents consists of code cleanup, unit tests and the introduction of
more sophisticated methods to do action space reduction with the :doc:`agents/teacher`.
Further it is possible to train the agent using `RLLib <https://www.ray.io/rllib>`_ which makes trying
out different RL algorithms easier.

Take a look at the :doc:`overview`, to get a feel of how the agent works and how each part of it can be
used.

.. toctree::
    :maxdepth: 1

    overview
    agents/teacher
    agents/tutor
    agents/junior
    agents/senior

    Generated API Documentation<api/curriculumagent>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
