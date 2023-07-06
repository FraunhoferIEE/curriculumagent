.. currentmodule:: curriculumagent.senior


Senior
=======

The *Senior* agent is the final stage of the Curriculum Agent and consist of a Deep Reinforcement Learning (DRL)
model based on the Proximal Policy Optimization (`PPO <https://arxiv.org/abs/1707.06347>`_.).
In the agent itself is trained with the same action set as the :mod:`~curriculumagent.tutor`, however
instead of a greedy approach the agents selects the actions based on the reward of the Grid2Op environment. Further,
in order to achieve a faster training result, the *Senior* agent receives the weights of the :doc:`junior`
agent for a warm start.

In the CurriculumAgent pipeline there exist two different implementations of the *Senior* agent, both based on the
PPO algorithm. The first approach is based on the
`Stable Baselines <https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html>`_ and can be executed
by :mod:`~curriculumagent.senior.senior_student`. The second option is based on the
`(RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_ framework, which can be found in the
:mod:`~curriculumagent.senior.senior_student_rllib`. While the first option implements a the PPO in a
straightforward way, the section option enables the usage of different hyperparameter searches,
such as Population Based Training `(PBT) <https://docs.ray.io/en/latest/tune/tutorials/tune-advanced-tutorial.html>`_
, or even the selection of a different DRL algorithm, e.g., see `Algorithms
<https://docs.ray.io/en/latest/rllib/rllib-algorithms.html>`_.

After training the PPO, the modle then only needs to be transfered to the submission directory in order
to have a completed agent.
